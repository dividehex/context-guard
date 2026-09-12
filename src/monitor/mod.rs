//! The monitor runs every signal over one event, persists what it learned,
//! and produces a health result. Signals are pure functions in the submodules;
//! this module owns the database round-trips.

pub mod context;
pub mod identifiers;
pub mod known_values;
pub mod repetition;
pub mod scoring;
pub mod text;
pub mod tools;

use std::sync::Arc;

use serde::Serialize;

use crate::config::Config;
use crate::database::repo::{NewAnomaly, NewEvent, NewHealth, NewToolCall, TurnUpdate};
use crate::database::Database;
use crate::metrics::Metrics;
use crate::telemetry::event::{ConversationEvent, EventKind, Message, Role};
use identifiers::{IdKind, Identifier};
use known_values::{KnownValue, ValueKind};
use scoring::{Signal, WindowAnomaly};
use text::sha256_hex;

#[derive(Debug, Clone, Serialize)]
pub struct Scored {
    pub conversation_id: String,
    pub model: String,
    pub turn: u32,
    pub health: u32,
    pub risk: u32,
    pub status: String,
    pub summary: String,
}

#[derive(Debug)]
pub enum Outcome {
    Duplicate,
    Recorded(EventKind),
    Scored(Scored),
}

struct Finding {
    signal: Signal,
    detail: String,
    dedupe_key: String,
}

#[derive(Clone)]
pub struct Monitor {
    db: Database,
    config: Arc<Config>,
    metrics: Arc<Metrics>,
}

impl Monitor {
    pub fn new(db: Database, config: Arc<Config>, metrics: Arc<Metrics>) -> Monitor {
        Monitor {
            db,
            config,
            metrics,
        }
    }

    pub async fn process(&self, event: &ConversationEvent) -> anyhow::Result<Outcome> {
        if self.db.event_exists(&event.event_id).await? {
            return Ok(Outcome::Duplicate);
        }
        self.db
            .touch_conversation(
                &event.conversation_id,
                event.conversation_id_source.as_str(),
                event.user_id.as_deref(),
                &event.model,
                event.timestamp,
            )
            .await?;

        if event.kind != EventKind::Chat {
            self.db
                .insert_event(self.new_event(event, None, None))
                .await?;
            return Ok(Outcome::Recorded(event.kind));
        }

        let conversation = self
            .db
            .get_conversation(&event.conversation_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("conversation vanished after upsert"))?;
        let turn = u32::try_from(conversation.turns).unwrap_or(0) + 1;
        let cid = event.conversation_id.as_str();

        // Messages not seen in the previous request of this conversation.
        let hashes: Vec<String> = event.messages.iter().map(message_hash).collect();
        let prev_count = usize::try_from(conversation.last_messages_count).unwrap_or(0);
        let prefix_matches = prev_count <= hashes.len()
            && conversation.last_messages_hash.as_deref()
                == Some(&sha256_hex(&hashes[..prev_count].join("\n")));
        let delta: &[Message] = if prefix_matches {
            &event.messages[prev_count..]
        } else {
            &event.messages
        };
        let full_hash = sha256_hex(&hashes.join("\n"));

        let new_messages_json = serde_json::to_string(delta).ok();
        let full_messages_json = if self.config.store_messages {
            serde_json::to_string(&event.messages).ok()
        } else {
            None
        };
        self.db
            .insert_event(self.new_event(
                event,
                Some(turn),
                Some((new_messages_json, full_messages_json)),
            ))
            .await?;

        let mut findings: Vec<Finding> = Vec::new();
        let ctx = context::assess(event.prompt_tokens, event.context_limit);
        self.learn_and_check_known_values(event, cid, turn, delta, &mut findings)
            .await?;
        self.learn_and_check_identifiers(event, cid, turn, delta, &mut findings)
            .await?;
        self.track_tools(event, cid, turn, delta, &mut findings)
            .await?;
        self.check_response_loop(event, cid, turn, &mut findings)
            .await?;

        for f in &findings {
            let inserted = self
                .db
                .insert_anomaly(NewAnomaly {
                    conversation_id: cid,
                    turn,
                    ts: event.timestamp,
                    signal: f.signal.as_str(),
                    penalty: self.config.penalties.for_signal(f.signal),
                    severity: f.signal.severity(),
                    detail: &f.detail,
                    dedupe_key: &f.dedupe_key,
                })
                .await?;
            if inserted {
                self.metrics.record_anomaly(&event.model, f.signal);
                tracing::info!(conversation = cid, turn, signal = f.signal.as_str(), detail = %f.detail, "anomaly recorded");
            }
        }

        let window_start = turn.saturating_sub(self.config.scoring.window_turns.saturating_sub(1));
        let window: Vec<WindowAnomaly> = self
            .db
            .anomalies_since_turn(cid, window_start)
            .await?
            .into_iter()
            .filter_map(|row| {
                Some(WindowAnomaly {
                    signal: Signal::parse(&row.signal)?,
                    penalty: u32::try_from(row.penalty).unwrap_or(0),
                    detail: row.detail,
                    turn: u32::try_from(row.turn).unwrap_or(0),
                })
            })
            .collect();

        let score = scoring::score(
            &ctx,
            &window,
            &self.config.penalties,
            &self.config.thresholds,
        );
        let summary = scoring::summary(&score, &ctx, &window);
        let reasons_json =
            serde_json::to_string(&score.reasons).unwrap_or_else(|_| "[]".to_string());

        self.db
            .insert_health(NewHealth {
                conversation_id: cid,
                turn,
                ts: event.timestamp,
                message_id: event.message_id.as_deref(),
                health: score.health,
                risk: score.risk,
                status: score.status.as_str(),
                context_percent: ctx.percent,
                prompt_tokens: ctx.prompt_tokens,
                context_limit: ctx.limit,
                reasons_json: &reasons_json,
                summary: &summary,
            })
            .await?;
        self.db
            .update_conversation_turn(TurnUpdate {
                conversation_id: cid,
                turns: turn,
                health: score.health,
                risk: score.risk,
                status: score.status.as_str(),
                context_percent: ctx.percent,
                prompt_tokens: ctx.prompt_tokens,
                context_limit: ctx.limit,
                messages_count: event.messages.len(),
                messages_hash: &full_hash,
            })
            .await?;
        self.metrics.record_score(
            &event.model,
            score.health,
            score.risk,
            ctx.percent.map(|p| p / 100.0),
        );

        Ok(Outcome::Scored(Scored {
            conversation_id: cid.to_string(),
            model: event.model.clone(),
            turn,
            health: score.health,
            risk: score.risk,
            status: score.status.as_str().to_string(),
            summary,
        }))
    }

    fn new_event<'a>(
        &self,
        event: &'a ConversationEvent,
        turn: Option<u32>,
        messages: Option<(Option<String>, Option<String>)>,
    ) -> NewEvent<'a> {
        let (new_messages_json, full_messages_json) = messages.unwrap_or((None, None));
        NewEvent {
            id: &event.event_id,
            conversation_id: &event.conversation_id,
            turn,
            kind: event.kind.as_str(),
            ts: event.timestamp,
            model: &event.model,
            message_id: event.message_id.as_deref(),
            prompt_tokens: event.prompt_tokens,
            completion_tokens: event.completion_tokens,
            context_limit: event.context_limit,
            response_text: event.response_text.as_deref(),
            new_messages_json,
            full_messages_json,
        }
    }

    async fn learn_and_check_known_values(
        &self,
        event: &ConversationEvent,
        cid: &str,
        turn: u32,
        delta: &[Message],
        findings: &mut Vec<Finding>,
    ) -> anyhow::Result<()> {
        for m in delta
            .iter()
            .filter(|m| matches!(m.role, Role::User | Role::Tool))
        {
            for v in known_values::extract(&m.content, &self.config.container_prefixes) {
                self.db
                    .upsert_known_value(
                        cid,
                        v.kind.as_str(),
                        &v.anchor,
                        &v.value,
                        m.role.as_str(),
                        turn,
                    )
                    .await?;
            }
        }
        let Some(response) = event.response_text.as_deref() else {
            return Ok(());
        };
        let registry: Vec<KnownValue> = self
            .db
            .known_values(cid)
            .await?
            .into_iter()
            .filter_map(|r| {
                Some(KnownValue {
                    kind: ValueKind::parse(&r.kind)?,
                    anchor: r.anchor,
                    value: r.value,
                })
            })
            .collect();
        if registry.is_empty() {
            return Ok(());
        }
        let claims = known_values::extract(response, &self.config.container_prefixes);
        for d in known_values::detect_drift(&claims, &registry) {
            let entity = if d.anchor.is_empty() {
                d.kind.as_str().to_string()
            } else {
                format!("{} of {}", d.kind.as_str(), d.anchor)
            };
            findings.push(Finding {
                signal: Signal::KnownValueDrift,
                detail: format!(
                    "assistant said {} {} but the conversation established {}",
                    entity, d.claimed, d.known
                ),
                dedupe_key: format!("kvd:{}:{}:{}", d.kind.as_str(), d.anchor, d.claimed),
            });
        }
        Ok(())
    }

    async fn learn_and_check_identifiers(
        &self,
        event: &ConversationEvent,
        cid: &str,
        turn: u32,
        delta: &[Message],
        findings: &mut Vec<Finding>,
    ) -> anyhow::Result<()> {
        self.db
            .upsert_identifier(cid, IdKind::Model.as_str(), &event.model, "system", turn)
            .await?;
        for m in delta {
            match m.role {
                Role::User | Role::Tool => {
                    for id in identifiers::extract(&m.content, &self.config.container_prefixes) {
                        self.db
                            .upsert_identifier(
                                cid,
                                id.kind.as_str(),
                                &id.value,
                                m.role.as_str(),
                                turn,
                            )
                            .await?;
                    }
                }
                Role::Assistant => {
                    for tc in &m.tool_calls {
                        self.db
                            .upsert_identifier(
                                cid,
                                IdKind::Tool.as_str(),
                                &tc.name,
                                "assistant",
                                turn,
                            )
                            .await?;
                    }
                }
                _ => {}
            }
        }
        for tc in &event.tool_calls {
            self.db
                .upsert_identifier(cid, IdKind::Tool.as_str(), &tc.name, "assistant", turn)
                .await?;
        }
        let Some(response) = event.response_text.as_deref() else {
            return Ok(());
        };
        let registry: Vec<Identifier> = self
            .db
            .identifiers(cid)
            .await?
            .into_iter()
            .filter_map(|r| {
                Some(Identifier {
                    kind: IdKind::parse(&r.kind)?,
                    value: r.value,
                })
            })
            .collect();
        let claims = identifiers::extract(response, &self.config.container_prefixes);
        for s in identifiers::detect_suspicious(&claims, &registry) {
            findings.push(Finding {
                signal: Signal::SuspiciousIdentifier,
                detail: format!(
                    "assistant mentioned {} which resembles the known {} {}",
                    s.claimed.value,
                    s.similar_to.kind.as_str(),
                    s.similar_to.value
                ),
                dedupe_key: format!("sid:{}", s.claimed.value),
            });
        }
        Ok(())
    }

    async fn track_tools(
        &self,
        event: &ConversationEvent,
        cid: &str,
        turn: u32,
        delta: &[Message],
        findings: &mut Vec<Finding>,
    ) -> anyhow::Result<()> {
        for orphan in tools::orphan_results(&event.messages) {
            findings.push(Finding {
                signal: Signal::ToolResultWithoutCall,
                detail: format!("tool result {orphan} has no matching tool call in the request"),
                dedupe_key: format!("orphan:{turn}:{orphan}"),
            });
        }
        for m in delta.iter().filter(|m| m.role == Role::Tool) {
            if let Some(call_id) = m.tool_call_id.as_deref() {
                let status = tools::result_status(&m.content);
                self.db
                    .record_tool_result(
                        cid,
                        call_id,
                        turn,
                        &sha256_hex(&m.content),
                        status.as_str(),
                    )
                    .await?;
            }
        }
        if let Some(response) = event.response_text.as_deref() {
            let known_ids = self.db.tool_call_ids(cid).await?;
            for id in tools::unknown_call_id_references(response, &known_ids) {
                findings.push(Finding {
                    signal: Signal::ToolCallIdReferenceUnknown,
                    detail: format!("assistant referenced tool call {id} which was never issued"),
                    dedupe_key: format!("ref:{id}"),
                });
            }
        }
        for (i, tc) in event.tool_calls.iter().enumerate() {
            let call_id = tc.id.clone().unwrap_or_else(|| format!("turn{turn}-{i}"));
            let key = repetition::tool_call_key(&tc.name, &tc.arguments);
            self.db
                .insert_tool_call(NewToolCall {
                    conversation_id: cid,
                    call_id: &call_id,
                    turn,
                    name: &tc.name,
                    key: &key,
                    args_json: &tc.arguments,
                    at: event.timestamp,
                })
                .await?;
        }
        if !event.tool_calls.is_empty() {
            let recent = self
                .db
                .recent_tool_keys(cid, repetition::TOOL_WINDOW as u32)
                .await?;
            for key in repetition::repeated_tool_calls(&recent) {
                let name = event
                    .tool_calls
                    .iter()
                    .find(|tc| repetition::tool_call_key(&tc.name, &tc.arguments) == key)
                    .map(|tc| tc.name.clone())
                    .unwrap_or_else(|| "tool".to_string());
                let occurrences = self.db.count_tool_key(cid, &key).await?;
                findings.push(Finding {
                    signal: Signal::RepeatedToolCall,
                    detail: format!(
                        "{name} called with identical arguments {} times",
                        occurrences
                    ),
                    dedupe_key: format!(
                        "rep:{key}:{}",
                        occurrences / repetition::TOOL_REPEAT_THRESHOLD as u32
                    ),
                });
            }
        }
        Ok(())
    }

    async fn check_response_loop(
        &self,
        event: &ConversationEvent,
        cid: &str,
        turn: u32,
        findings: &mut Vec<Finding>,
    ) -> anyhow::Result<()> {
        let Some(response) = event.response_text.as_deref() else {
            return Ok(());
        };
        let previous = self
            .db
            .recent_responses(cid, turn, repetition::RESPONSE_HISTORY as u32)
            .await?;
        if let Some(similarity) = repetition::response_loop(response, &previous) {
            findings.push(Finding {
                signal: Signal::ResponseLoop,
                detail: format!(
                    "response is {:.0}% similar to recent responses",
                    similarity * 100.0
                ),
                dedupe_key: format!("loop:{turn}"),
            });
        }
        Ok(())
    }
}

fn message_hash(m: &Message) -> String {
    let tool_calls: Vec<String> = m
        .tool_calls
        .iter()
        .map(|tc| {
            format!(
                "{}:{}:{}",
                tc.id.as_deref().unwrap_or(""),
                tc.name,
                tc.arguments
            )
        })
        .collect();
    sha256_hex(&format!(
        "{}\u{1}{}\u{1}{}\u{1}{}",
        m.role.as_str(),
        m.content,
        m.tool_call_id.as_deref().unwrap_or(""),
        tool_calls.join("\u{2}")
    ))
}
