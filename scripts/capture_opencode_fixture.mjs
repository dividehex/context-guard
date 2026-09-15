// Capture a real opencode session as a Context Guard fixture.
//
// Reads the running TUI's server registration exactly like the CLI does
// (state dir server.json + password), pulls the session's flat v2 messages
// over the same HTTP API the SDK client uses, converts them to the
// canonical `{info, parts}` ingest record shape with the shipper's collect(),
// and writes `tests/fixtures/opencode-capture-<session>.json`.
//
// Usage:
//   node scripts/capture_opencode_fixture.mjs <session-id> [--context-limit N] [--out path]
//
// Env overrides (for testing the script without a live TUI):
//   OPENCODE_BASE_URL=http://127.0.0.1:PORT   skip the state-dir discovery + auth
//   PKG_TMPDIR=dir  (opens safe temp dirs for tests)
//
// This script is a contributor tool, not part of the Docker image and not
// covered by the CI node:test run (it needs a live opencode server).

import { readFileSync, writeFileSync } from "node:fs"
import path from "node:path"
import os from "node:os"

import { collect } from "../opencode/context_guard_shipper.mjs"

function fail(message) {
  console.error(`capture: ${message}`)
  process.exit(1)
}

function stateDir() {
  const base = process.env.XDG_STATE_HOME ?? path.join(os.homedir(), ".local", "state")
  return path.join(base, "opencode")
}

function credentials() {
  const dir = stateDir()
  try {
    const registration = JSON.parse(readFileSync(path.join(dir, "server.json"), "utf8"))
    const password = readFileSync(path.join(dir, "password"), "utf8").trim()
    const token = Buffer.from(`opencode:${password}`).toString("base64")
    return { url: registration.url, headers: { Authorization: `Basic ${token}` } }
  } catch (cause) {
    fail(`cannot read opencode server credentials from ${dir} (is the TUI running?): ${cause.message}`)
  }
}

function redact(text) {
  return text
    .replace(/sk-[A-Za-z0-9_-]{16,}/g, "sk-REDACTED")
    .replace(/ghp_[A-Za-z0-9]{30,}/g, "ghp_REDACTED")
    .replace(/github_pat_[A-Za-z0-9_]{20,}/g, "github_pat_REDACTED")
}

export async function main() {
  const args = process.argv.slice(2)
  const sessionID = args[0]
  const contextLimitArg = args.indexOf("--context-limit")
  const outArg = args.indexOf("--out")
  if (!sessionID) fail("expected <session-id> argument")

  const contextLimit = contextLimitArg >= 0 ? Number(args[contextLimitArg + 1]) || null : null
  const out = outArg >= 0 ? args[outArg + 1] : undefined

  const env = process.env.OPENCODE_BASE_URL
    ? { url: process.env.OPENCODE_BASE_URL, headers: {} }
    : credentials()

  const response = await fetch(`${env.url}/session/${encodeURIComponent(sessionID)}/message?limit=1000`, {
    headers: { accept: "application/json", ...env.headers },
  })
  if (!response.ok) fail(`GET session/message -> HTTP ${response.status}`)
  const body = await response.json()
  if (!Array.isArray(body.data)) fail("expected { data: SessionMessage[] } response body")

  const { records } = collect(body.data)
  if (!records || records.length === 0) fail("no convertible messages in session")

  const serialized = JSON.stringify(
    { session_id: sessionID, context_limit: contextLimit, records },
    (key, value) => (typeof value === "string" ? redact(value) : value),
    2,
  )

  const target =
    out ?? path.join(import.meta.dirname, "..", "tests", "fixtures", `opencode-capture-${sessionID.split("_").pop() ?? sessionID}.json`)
  writeFileSync(target, serialized + "\n")
  console.error(`wrote ${target}: ${records.length} records, context_limit ${contextLimit ?? "null"}`)
  console.error("review the file for tokens/paths before committing; run `git diff --cached` and redact by hand")
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((cause) => fail(cause.message))
}