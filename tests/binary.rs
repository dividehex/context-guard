//! The real binary: startup, the healthcheck subcommand, and the optional
//! metrics-only listener, all over plain TCP.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn http_get(addr: &str, path: &str) -> Option<(u16, String)> {
    let mut s = TcpStream::connect_timeout(&addr.parse().ok()?, Duration::from_millis(500)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    write!(s, "GET {path} HTTP/1.0\r\nHost: localhost\r\n\r\n").ok()?;
    let mut out = String::new();
    s.read_to_string(&mut out).ok()?;
    let status: u16 = out.split_whitespace().nth(1)?.parse().ok()?;
    let body = out
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Some((status, body))
}

fn wait_ready(addr: &str) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        if matches!(http_get(addr, "/healthz"), Some((200, _))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("service at {addr} did not become ready");
}

struct Service {
    child: Child,
    api: String,
    metrics: String,
    dir: tempfile::TempDir,
}

fn start() -> Service {
    let dir = tempfile::tempdir().unwrap();
    let api = format!("127.0.0.1:{}", free_port());
    let metrics = format!("127.0.0.1:{}", free_port());
    let child = Command::new(env!("CARGO_BIN_EXE_context-guard"))
        .env("CONTEXT_GUARD_LISTEN", &api)
        .env("CONTEXT_GUARD_METRICS_LISTEN", &metrics)
        .env("CONTEXT_GUARD_DATABASE", dir.path().join("cg.db"))
        .env("CONTEXT_GUARD_LOG_JSON", "true")
        .env("RUST_LOG", "info")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_ready(&api);
    Service {
        child,
        api,
        metrics,
        dir,
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
    }
}

#[test]
fn binary_serves_api_and_metrics_listener_and_healthcheck_passes() {
    let svc = start();
    let (status, body) = http_get(&svc.api, "/healthz").unwrap();
    assert_eq!(status, 200);
    assert!(body.contains("\"database\":\"ok\""), "{body}");
    assert!(svc.dir.path().join("cg.db").exists());

    // The metrics listener serves only /metrics and /healthz.
    let (status, body) = http_get(&svc.metrics, "/metrics").unwrap();
    assert_eq!(status, 200);
    assert!(body.contains("context_guard_queue_depth"));
    assert_eq!(http_get(&svc.metrics, "/healthz").unwrap().0, 200);
    assert_eq!(
        http_get(&svc.metrics, "/api/v1/conversations").unwrap().0,
        404
    );
    assert_eq!(http_get(&svc.api, "/api/v1/conversations").unwrap().0, 200);

    let ok = Command::new(env!("CARGO_BIN_EXE_context-guard"))
        .arg("healthcheck")
        .env("CONTEXT_GUARD_LISTEN", &svc.api)
        .env("CONTEXT_GUARD_DATABASE", svc.dir.path().join("cg.db"))
        .status()
        .unwrap();
    assert!(
        ok.success(),
        "healthcheck must exit 0 against a live service"
    );
}

#[test]
fn healthcheck_fails_when_nothing_listens_and_bad_config_exits_nonzero() {
    let dead = format!("127.0.0.1:{}", free_port());
    let status = Command::new(env!("CARGO_BIN_EXE_context-guard"))
        .arg("healthcheck")
        .env("CONTEXT_GUARD_LISTEN", &dead)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!status.success());

    let status = Command::new(env!("CARGO_BIN_EXE_context-guard"))
        .env("CONTEXT_GUARD_RETENTION_DAYS", "soon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "an invalid configuration must not start the service"
    );

    let status = Command::new(env!("CARGO_BIN_EXE_context-guard"))
        .env("CONTEXT_GUARD_LISTEN", format!("127.0.0.1:{}", free_port()))
        .env(
            "CONTEXT_GUARD_DATABASE",
            "/proc/definitely/not/writable/cg.db",
        )
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(
        !status.success(),
        "an unopenable database must exit non-zero so the container restarts"
    );
}
