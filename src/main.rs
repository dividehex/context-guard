//! Context Guard: a deterministic, out-of-band health monitor for LLM
//! conversations. It observes LiteLLM telemetry, records it, scores it, and
//! reports it. It is never in the inference path.

use std::sync::Arc;
use std::time::Instant;

use context_guard::{api, config::Config, database, metrics, monitor, worker};
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        return healthcheck();
    }

    let config = Arc::new(Config::load()?);
    init_tracing(config.log_json);
    tracing::info!(
        listen = %config.listen,
        database = %config.database.display(),
        retention_days = config.retention_days,
        version = env!("CARGO_PKG_VERSION"),
        "context guard starting"
    );

    if let Some(parent) = config.database.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let db = database::Database::connect(&config.database).await?;
    let metrics = Arc::new(metrics::Metrics::new());
    let (tx, rx) = mpsc::channel::<worker::Batch>(config.queue_size);

    let monitor = monitor::Monitor::new(db.clone(), config.clone(), metrics.clone());
    tokio::spawn(worker::run(rx, monitor, config.clone(), metrics.clone()));
    tokio::spawn(database::retention_loop(db.clone(), config.retention_days));

    let state = api::AppState {
        db,
        tx,
        metrics,
        config: config.clone(),
        started: Instant::now(),
    };

    if let Some(addr) = config.metrics_listen.clone() {
        let router = api::metrics_router(state.clone());
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!(%addr, "metrics listener ready");
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, router).await {
                tracing::error!(error = %e, "metrics listener failed");
            }
        });
    }

    let listener = tokio::net::TcpListener::bind(&config.listen).await?;
    tracing::info!(addr = %config.listen, "api listener ready");
    axum::serve(listener, api::router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("context guard stopped");
    Ok(())
}

fn init_tracing(json: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("signal handler");
        tokio::select! {
            _ = ctrl_c => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    ctrl_c.await.ok();
}

/// `context-guard healthcheck`: GET /healthz over plain TCP so the runtime
/// image needs no curl. Exit 0 on HTTP 200.
fn healthcheck() -> anyhow::Result<()> {
    use std::io::{Read, Write};
    let config = Config::load()?;
    let addr = config
        .listen
        .replace("0.0.0.0", "127.0.0.1")
        .replace("[::]", "[::1]");
    let mut stream = std::net::TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|e| anyhow::anyhow!("bad listen address {addr}: {e}"))?,
        std::time::Duration::from_secs(2),
    )?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
    stream.write_all(b"GET /healthz HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok();
    anyhow::ensure!(
        response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200"),
        "unhealthy: {}",
        response.lines().next().unwrap_or("")
    );
    Ok(())
}
