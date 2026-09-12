# syntax=docker/dockerfile:1
# Multi-stage build: compile in the official Rust image, run in a slim Debian
# image as an unprivileged user. SQLite is linked statically by sqlx.

FROM rust:1-bookworm AS builder
WORKDIR /src

# Cache dependencies separately from the source.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs && echo '' > src/lib.rs \
    && cargo build --release --locked 2>/dev/null || true

COPY migrations ./migrations
COPY src ./src
RUN touch src/main.rs src/lib.rs && cargo build --release --locked --bin context-guard

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 context-guard \
    && useradd --uid 10001 --gid 10001 --home /data --shell /usr/sbin/nologin context-guard \
    && mkdir -p /data && chown context-guard:context-guard /data

COPY --from=builder /src/target/release/context-guard /usr/local/bin/context-guard

USER context-guard
VOLUME ["/data"]
EXPOSE 7432
ENV CONTEXT_GUARD_LISTEN=0.0.0.0:7432 \
    CONTEXT_GUARD_DATABASE=/data/context-guard.db \
    RUST_LOG=info

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["context-guard", "healthcheck"]

ENTRYPOINT ["context-guard"]
