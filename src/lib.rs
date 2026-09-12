//! Context Guard library: everything the binary wires together, exposed so
//! integration tests can drive the HTTP router in-process.

pub mod api;
pub mod config;
pub mod database;
pub mod metrics;
pub mod monitor;
pub mod telemetry;
pub mod worker;
