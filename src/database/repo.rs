//! Typed queries. Every function is one statement or a tight sequence; the
//! monitor decides what to do with the rows.

use chrono::{DateTime, Utc};
use sqlx::FromRow;

use super::Database;

type Result<T> = std::result::Result<T, sqlx::Error>;

pub(crate) fn ts(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

#[derive(Debug, Clone, FromRow)]
pub struct ConversationRow {
    pub id: String,
    pub id_source: String,
    pub user_id: Option<String>,
    pub model: String,
    pub first_seen: String,
    pub last_seen: String,
    pub turns: i64,
    pub prompts: i64,
    pub last_health: Option<i64>,
    pub last_risk: Option<i64>,
    pub last_status: Option<String>,
    pub last_context_percent: Option<f64>,
    pub last_prompt_tokens: Option<i64>,
    pub last_context_limit: Option<i64>,
    pub last_messages_count: i64,
    pub last_messages_hash: Option<String>,
}

#[derive(Debug, Clone, FromRow)]
pub struct HealthRow {
    pub turn: i64,
    pub prompt: i64,
    pub ts: String,
    pub message_id: Option<String>,
    pub health: i64,
    pub risk: i64,
    pub status: String,
    pub context_percent: Option<f64>,
    pub prompt_tokens: Option<i64>,
    pub context_limit: Option<i64>,
    pub reasons_json: String,
    pub summary: String,
}

#[derive(Debug, Clone, FromRow)]
pub struct AnomalyRow {
    pub turn: i64,
    pub prompt: i64,
    pub ts: String,
    pub signal: String,
    pub penalty: i64,
    pub severity: String,
    pub detail: String,
}

#[derive(Debug, Clone, FromRow)]
pub struct KnownValueRow {
    pub kind: String,
    pub anchor: String,
    pub value: String,
}

#[derive(Debug, Clone, FromRow)]
pub struct IdentifierRow {
    pub kind: String,
    pub value: String,
}

#[derive(Debug, Clone, FromRow)]
pub struct StatusCount {
    pub status: String,
    pub count: i64,
}

pub struct NewEvent<'a> {
    pub id: &'a str,
    pub conversation_id: &'a str,
    pub turn: Option<u32>,
    pub kind: &'a str,
    pub ts: DateTime<Utc>,
    pub model: &'a str,
    pub message_id: Option<&'a str>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub context_limit: Option<u64>,
    pub response_text: Option<&'a str>,
    pub new_messages_json: Option<String>,
    pub full_messages_json: Option<String>,
}

pub struct TurnUpdate<'a> {
    pub conversation_id: &'a str,
    pub turns: u32,
    pub prompts: u32,
    pub health: u32,
    pub risk: u32,
    pub status: &'a str,
    pub context_percent: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub context_limit: Option<u64>,
    pub messages_count: usize,
    pub messages_hash: &'a str,
}

pub struct NewHealth<'a> {
    pub conversation_id: &'a str,
    pub turn: u32,
    pub prompt: u32,
    pub ts: DateTime<Utc>,
    pub message_id: Option<&'a str>,
    pub health: u32,
    pub risk: u32,
    pub status: &'a str,
    pub context_percent: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub context_limit: Option<u64>,
    pub reasons_json: &'a str,
    pub summary: &'a str,
}

pub struct NewToolCall<'a> {
    pub conversation_id: &'a str,
    pub call_id: &'a str,
    pub turn: u32,
    pub name: &'a str,
    /// Operation key: hash of name + canonical arguments.
    pub key: &'a str,
    pub args_json: &'a str,
    pub at: DateTime<Utc>,
}

pub struct NewAnomaly<'a> {
    pub conversation_id: &'a str,
    pub turn: u32,
    pub prompt: u32,
    pub ts: DateTime<Utc>,
    pub signal: &'a str,
    pub penalty: u32,
    pub severity: &'a str,
    pub detail: &'a str,
    pub dedupe_key: &'a str,
}

fn opt_i64(v: Option<u64>) -> Option<i64> {
    v.map(|n| i64::try_from(n).unwrap_or(i64::MAX))
}

impl Database {
    pub async fn event_exists(&self, id: &str) -> Result<bool> {
        let row: Option<(i64,)> = sqlx::query_as("SELECT 1 FROM conversation_events WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await?;
        Ok(row.is_some())
    }

    /// Create or refresh a conversation. The stored model follows chat turns
    /// only (`is_turn`): background tasks often run on a different model and
    /// must not relabel the chat.
    pub async fn touch_conversation(
        &self,
        id: &str,
        id_source: &str,
        user_id: Option<&str>,
        model: &str,
        is_turn: bool,
        seen: DateTime<Utc>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO conversations (id, id_source, user_id, model, first_seen, last_seen)
             VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
               last_seen = MAX(last_seen, excluded.last_seen),
               model = COALESCE(?, conversations.model),
               user_id = COALESCE(excluded.user_id, conversations.user_id)",
        )
        .bind(id)
        .bind(id_source)
        .bind(user_id)
        .bind(model)
        .bind(ts(seen))
        .bind(ts(seen))
        .bind(is_turn.then_some(model))
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn get_conversation(&self, id: &str) -> Result<Option<ConversationRow>> {
        sqlx::query_as("SELECT * FROM conversations WHERE id = ?")
            .bind(id)
            .fetch_optional(self.pool())
            .await
    }

    pub async fn insert_event(&self, e: NewEvent<'_>) -> Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO conversation_events
             (id, conversation_id, turn, kind, ts, model, message_id, prompt_tokens, completion_tokens,
              context_limit, response_text, new_messages_json, full_messages_json)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(e.id)
        .bind(e.conversation_id)
        .bind(e.turn.map(i64::from))
        .bind(e.kind)
        .bind(ts(e.ts))
        .bind(e.model)
        .bind(e.message_id)
        .bind(opt_i64(e.prompt_tokens))
        .bind(opt_i64(e.completion_tokens))
        .bind(opt_i64(e.context_limit))
        .bind(e.response_text)
        .bind(e.new_messages_json)
        .bind(e.full_messages_json)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn update_conversation_turn(&self, u: TurnUpdate<'_>) -> Result<()> {
        sqlx::query(
            "UPDATE conversations SET turns = ?, prompts = ?, last_health = ?, last_risk = ?, last_status = ?,
             last_context_percent = ?, last_prompt_tokens = ?, last_context_limit = ?,
             last_messages_count = ?, last_messages_hash = ? WHERE id = ?",
        )
        .bind(i64::from(u.turns))
        .bind(i64::from(u.prompts))
        .bind(i64::from(u.health))
        .bind(i64::from(u.risk))
        .bind(u.status)
        .bind(u.context_percent)
        .bind(opt_i64(u.prompt_tokens))
        .bind(opt_i64(u.context_limit))
        .bind(u.messages_count as i64)
        .bind(u.messages_hash)
        .bind(u.conversation_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn insert_health(&self, h: NewHealth<'_>) -> Result<()> {
        sqlx::query(
            "INSERT INTO health_results (conversation_id, turn, prompt, ts, message_id, health, risk, status,
             context_percent, prompt_tokens, context_limit, reasons_json, summary)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(h.conversation_id)
        .bind(i64::from(h.turn))
        .bind(i64::from(h.prompt))
        .bind(ts(h.ts))
        .bind(h.message_id)
        .bind(i64::from(h.health))
        .bind(i64::from(h.risk))
        .bind(h.status)
        .bind(h.context_percent)
        .bind(opt_i64(h.prompt_tokens))
        .bind(opt_i64(h.context_limit))
        .bind(h.reasons_json)
        .bind(h.summary)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Assistant responses of the most recent chat turns before `before_turn`, oldest first.
    pub async fn recent_responses(
        &self,
        conversation_id: &str,
        before_turn: u32,
        limit: u32,
    ) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT response_text FROM conversation_events
             WHERE conversation_id = ? AND kind = 'chat' AND turn < ? AND response_text IS NOT NULL
             ORDER BY turn DESC LIMIT ?",
        )
        .bind(conversation_id)
        .bind(i64::from(before_turn))
        .bind(i64::from(limit))
        .fetch_all(self.pool())
        .await?;
        Ok(rows.into_iter().rev().map(|(t,)| t).collect())
    }

    pub async fn insert_tool_call(&self, t: NewToolCall<'_>) -> Result<()> {
        sqlx::query(
            "INSERT INTO tool_calls (conversation_id, call_id, turn, name, args_hash, args_json, ts)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(t.conversation_id)
        .bind(t.call_id)
        .bind(i64::from(t.turn))
        .bind(t.name)
        .bind(t.key)
        .bind(t.args_json)
        .bind(ts(t.at))
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Attach a result to the newest unresolved call with this id.
    pub async fn record_tool_result(
        &self,
        conversation_id: &str,
        call_id: &str,
        result_turn: u32,
        result_hash: &str,
        status: &str,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE tool_calls SET result_turn = ?, result_hash = ?, result_status = ?
             WHERE id = (SELECT id FROM tool_calls WHERE conversation_id = ? AND call_id = ? AND result_turn IS NULL
                         ORDER BY id DESC LIMIT 1)",
        )
        .bind(i64::from(result_turn))
        .bind(result_hash)
        .bind(status)
        .bind(conversation_id)
        .bind(call_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Operation keys of the last `limit` tool calls, oldest first.
    pub async fn recent_tool_keys(&self, conversation_id: &str, limit: u32) -> Result<Vec<String>> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT args_hash FROM tool_calls WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        )
        .bind(conversation_id)
        .bind(i64::from(limit))
        .fetch_all(self.pool())
        .await?;
        Ok(rows.into_iter().rev().map(|(k,)| k).collect())
    }

    pub async fn count_tool_key(&self, conversation_id: &str, key: &str) -> Result<u32> {
        let (n,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM tool_calls WHERE conversation_id = ? AND args_hash = ?",
        )
        .bind(conversation_id)
        .bind(key)
        .fetch_one(self.pool())
        .await?;
        Ok(u32::try_from(n).unwrap_or(u32::MAX))
    }

    pub async fn tool_call_ids(&self, conversation_id: &str) -> Result<Vec<String>> {
        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT DISTINCT call_id FROM tool_calls WHERE conversation_id = ?")
                .bind(conversation_id)
                .fetch_all(self.pool())
                .await?;
        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    pub async fn known_values(&self, conversation_id: &str) -> Result<Vec<KnownValueRow>> {
        sqlx::query_as(
            "SELECT kind, anchor, value FROM known_values WHERE conversation_id = ? ORDER BY id",
        )
        .bind(conversation_id)
        .fetch_all(self.pool())
        .await
    }

    pub async fn upsert_known_value(
        &self,
        conversation_id: &str,
        kind: &str,
        anchor: &str,
        value: &str,
        source_role: &str,
        turn: u32,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO known_values (conversation_id, kind, anchor, value, source_role, first_turn, last_turn)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(conversation_id, kind, anchor, value) DO UPDATE SET
               last_turn = excluded.last_turn, occurrences = occurrences + 1",
        )
        .bind(conversation_id)
        .bind(kind)
        .bind(anchor)
        .bind(value)
        .bind(source_role)
        .bind(i64::from(turn))
        .bind(i64::from(turn))
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn identifiers(&self, conversation_id: &str) -> Result<Vec<IdentifierRow>> {
        sqlx::query_as("SELECT kind, value FROM identifiers WHERE conversation_id = ? ORDER BY id")
            .bind(conversation_id)
            .fetch_all(self.pool())
            .await
    }

    pub async fn upsert_identifier(
        &self,
        conversation_id: &str,
        kind: &str,
        value: &str,
        source_role: &str,
        turn: u32,
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO identifiers (conversation_id, kind, value, source_role, first_turn)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(conversation_id)
        .bind(kind)
        .bind(value)
        .bind(source_role)
        .bind(i64::from(turn))
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Returns true when the anomaly was new (not deduplicated).
    pub async fn insert_anomaly(&self, a: NewAnomaly<'_>) -> Result<bool> {
        let result = sqlx::query(
            "INSERT OR IGNORE INTO anomalies (conversation_id, turn, prompt, ts, signal, penalty, severity, detail, dedupe_key)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(a.conversation_id)
        .bind(i64::from(a.turn))
        .bind(i64::from(a.prompt))
        .bind(ts(a.ts))
        .bind(a.signal)
        .bind(i64::from(a.penalty))
        .bind(a.severity)
        .bind(a.detail)
        .bind(a.dedupe_key)
        .execute(self.pool())
        .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Anomalies inside the scoring window, which is counted in prompts.
    pub async fn anomalies_since_prompt(
        &self,
        conversation_id: &str,
        min_prompt: u32,
    ) -> Result<Vec<AnomalyRow>> {
        sqlx::query_as(
            "SELECT turn, prompt, ts, signal, penalty, severity, detail FROM anomalies
             WHERE conversation_id = ? AND prompt >= ? ORDER BY turn, id",
        )
        .bind(conversation_id)
        .bind(i64::from(min_prompt))
        .fetch_all(self.pool())
        .await
    }

    pub async fn anomalies(&self, conversation_id: &str) -> Result<Vec<AnomalyRow>> {
        self.anomalies_since_prompt(conversation_id, 0).await
    }

    pub async fn list_conversations(
        &self,
        limit: u32,
        status: Option<&str>,
    ) -> Result<Vec<ConversationRow>> {
        match status {
            Some(s) => sqlx::query_as(
                "SELECT * FROM conversations WHERE last_status = ? ORDER BY last_seen DESC LIMIT ?",
            )
            .bind(s)
            .bind(i64::from(limit))
            .fetch_all(self.pool())
            .await,
            None => {
                sqlx::query_as("SELECT * FROM conversations ORDER BY last_seen DESC LIMIT ?")
                    .bind(i64::from(limit))
                    .fetch_all(self.pool())
                    .await
            }
        }
    }

    pub async fn latest_health(&self, conversation_id: &str) -> Result<Option<HealthRow>> {
        sqlx::query_as(
            "SELECT turn, prompt, ts, message_id, health, risk, status, context_percent, prompt_tokens, context_limit, reasons_json, summary
             FROM health_results WHERE conversation_id = ? ORDER BY turn DESC, id DESC LIMIT 1",
        )
        .bind(conversation_id)
        .fetch_optional(self.pool())
        .await
    }

    /// Latest result for an Open WebUI message (a reply may span several tool iterations).
    pub async fn health_for_message(
        &self,
        conversation_id: &str,
        message_id: &str,
    ) -> Result<Option<HealthRow>> {
        sqlx::query_as(
            "SELECT turn, prompt, ts, message_id, health, risk, status, context_percent, prompt_tokens, context_limit, reasons_json, summary
             FROM health_results WHERE conversation_id = ? AND message_id = ? ORDER BY turn DESC, id DESC LIMIT 1",
        )
        .bind(conversation_id)
        .bind(message_id)
        .fetch_optional(self.pool())
        .await
    }

    pub async fn health_after(
        &self,
        conversation_id: &str,
        after: DateTime<Utc>,
    ) -> Result<Option<HealthRow>> {
        sqlx::query_as(
            "SELECT turn, prompt, ts, message_id, health, risk, status, context_percent, prompt_tokens, context_limit, reasons_json, summary
             FROM health_results WHERE conversation_id = ? AND ts >= ? ORDER BY turn DESC, id DESC LIMIT 1",
        )
        .bind(conversation_id)
        .bind(ts(after))
        .fetch_optional(self.pool())
        .await
    }

    pub async fn health_history(
        &self,
        conversation_id: &str,
        limit: u32,
    ) -> Result<Vec<HealthRow>> {
        let rows: Vec<HealthRow> = sqlx::query_as(
            "SELECT turn, prompt, ts, message_id, health, risk, status, context_percent, prompt_tokens, context_limit, reasons_json, summary
             FROM health_results WHERE conversation_id = ? ORDER BY turn DESC, id DESC LIMIT ?",
        )
        .bind(conversation_id)
        .bind(i64::from(limit))
        .fetch_all(self.pool())
        .await?;
        Ok(rows.into_iter().rev().collect())
    }

    pub async fn status_counts_since(&self, since: DateTime<Utc>) -> Result<Vec<StatusCount>> {
        sqlx::query_as(
            "SELECT last_status AS status, COUNT(*) AS count FROM conversations
             WHERE last_status IS NOT NULL AND last_seen >= ? GROUP BY last_status",
        )
        .bind(ts(since))
        .fetch_all(self.pool())
        .await
    }
}
