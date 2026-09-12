//! SQLite persistence: connection, migrations, retention.

use std::path::Path;
use std::str::FromStr;
use std::time::Duration;

use chrono::{DateTime, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;

pub mod repo;

#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn connect(path: &Path) -> anyhow::Result<Database> {
        let url = format!("sqlite://{}", path.display());
        let options = SqliteConnectOptions::from_str(&url)?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(5))
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(options)
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Database { pool })
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Cheap liveness probe for `/healthz`.
    pub async fn ping(&self) -> bool {
        sqlx::query("SELECT 1").execute(&self.pool).await.is_ok()
    }

    /// Delete conversations (and, by cascade, everything they own) not seen
    /// since `cutoff`. Returns the number of conversations removed.
    pub async fn purge_before(&self, cutoff: DateTime<Utc>) -> anyhow::Result<u64> {
        let result = sqlx::query("DELETE FROM conversations WHERE last_seen < ?")
            .bind(cutoff.to_rfc3339())
            .execute(&self.pool)
            .await?;
        if result.rows_affected() > 0 {
            sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
                .execute(&self.pool)
                .await
                .ok();
        }
        Ok(result.rows_affected())
    }
}

/// Hourly retention sweep. Never exits; failures are logged.
pub async fn retention_loop(db: Database, retention_days: u32) {
    let interval = Duration::from_secs(3600);
    loop {
        let cutoff = Utc::now() - chrono::Duration::days(i64::from(retention_days));
        match db.purge_before(cutoff).await {
            Ok(0) => tracing::debug!("retention sweep: nothing to purge"),
            Ok(n) => tracing::info!(purged = n, "retention sweep purged stale conversations"),
            Err(e) => tracing::warn!(error = %e, "retention sweep failed"),
        }
        tokio::time::sleep(interval).await;
    }
}
