//! Real LiteLLM v1.94.1 `generic_api` batches captured from the running stack
//! (two turns of one chat sent through the proxy with Open WebUI-style headers).

use context_guard::config::Config;
use context_guard::telemetry::event::{EventKind, IdSource, Role};
use context_guard::telemetry::litellm::{normalize, split_body};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn real_non_streaming_turn_normalizes() {
    let payloads = split_body(&fixture("litellm-v1.94.1-turn1.json")).unwrap();
    assert_eq!(payloads.len(), 1);
    let ev = normalize(&payloads[0], &Config::default()).unwrap();
    assert_eq!(ev.conversation_id, "cg-smoke-1789217539");
    assert_eq!(ev.conversation_id_source, IdSource::ChatTag);
    assert_eq!(ev.user_id.as_deref(), Some("user-smoke"));
    assert_eq!(ev.message_id.as_deref(), Some("msg-smoke-1"));
    assert_eq!(ev.model, "qwen3-30b-a3b");
    assert_eq!(ev.kind, EventKind::Chat);
    assert_eq!(ev.prompt_tokens, Some(44));
    assert_eq!(ev.completion_tokens, Some(13));
    assert_eq!(
        ev.context_limit,
        Some(122_880),
        "max_input_tokens comes from LiteLLM's model_info"
    );
    assert_eq!(ev.messages.len(), 1);
    assert_eq!(ev.messages[0].role, Role::User);
    assert_eq!(
        ev.response_text.as_deref(),
        Some("Llama.cpp is using port 8080.")
    );
    assert!(ev.tool_calls.is_empty());
    assert_eq!(ev.timestamp.timestamp(), 1_789_217_570);
}

#[test]
fn real_streaming_turn_normalizes() {
    let payloads = split_body(&fixture("litellm-v1.94.1-turn2.json")).unwrap();
    let ev = normalize(&payloads[0], &Config::default()).unwrap();
    assert_eq!(ev.stream, Some(true));
    assert_eq!(ev.kind, EventKind::Chat);
    assert_eq!(ev.message_id.as_deref(), Some("msg-smoke-2"));
    assert_eq!(ev.messages.len(), 3);
    assert_eq!(ev.messages[1].role, Role::Assistant);
    assert!(ev.response_text.as_deref().unwrap().contains("port 8000"));
}
