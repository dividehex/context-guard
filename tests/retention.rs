use chrono::{Duration, Utc};
use context_guard::database::repo::{NewAnomaly, NewEvent};
use context_guard::database::Database;

#[tokio::test]
async fn purge_removes_stale_conversations_and_everything_they_own() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::connect(&dir.path().join("r.db")).await.unwrap();
    let now = Utc::now();
    let old_seen = now - Duration::days(40);

    db.touch_conversation("old", "chat_tag", None, "m", true, old_seen)
        .await
        .unwrap();
    db.insert_event(NewEvent {
        id: "e-old",
        conversation_id: "old",
        turn: Some(1),
        kind: "chat",
        ts: old_seen,
        model: "m",
        message_id: None,
        prompt_tokens: None,
        completion_tokens: None,
        context_limit: None,
        response_text: Some("x"),
        new_messages_json: None,
        full_messages_json: None,
    })
    .await
    .unwrap();
    db.insert_anomaly(NewAnomaly {
        conversation_id: "old",
        turn: 1,
        prompt: 1,
        ts: old_seen,
        signal: "response_loop",
        penalty: 5,
        severity: "low",
        detail: "d",
        dedupe_key: "k",
    })
    .await
    .unwrap();
    db.upsert_known_value("old", "port", "", "8080", "user", 1)
        .await
        .unwrap();
    db.touch_conversation("fresh", "chat_tag", None, "m", true, now)
        .await
        .unwrap();

    let purged = db.purge_before(now - Duration::days(30)).await.unwrap();
    assert_eq!(purged, 1);
    assert!(db.get_conversation("old").await.unwrap().is_none());
    assert!(
        db.anomalies("old").await.unwrap().is_empty(),
        "cascade removed the anomalies"
    );
    assert!(
        db.known_values("old").await.unwrap().is_empty(),
        "cascade removed the known values"
    );
    assert!(
        !db.event_exists("e-old").await.unwrap(),
        "cascade removed the events"
    );
    assert!(db.get_conversation("fresh").await.unwrap().is_some());

    assert_eq!(
        db.purge_before(now - Duration::days(30)).await.unwrap(),
        0,
        "idempotent"
    );
    assert!(db.ping().await);
}
