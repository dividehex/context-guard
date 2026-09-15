// Context Guard opencode shipper: the pure, dependency-free half of the
// opencode integration. It owns everything outside the TUI: where the
// per-session cursor lives, how a session's messages become an ingest body,
// how the body is posted, how health is fetched, and how the one-line status
// is rendered. Node >= 18 (fetch, node:test). The TUI module in
// context_guard.tui.tsx is the only consumer; tests drive this file directly.
//
// Session messages reach this module as one of two shapes:
//
//   - the `{ info, parts }` records the opencode SDK actually returns
//     (`/session/{id}/message` and, via the TUI plugin, `api.state.session
//     .messages()` + `api.state.part(messageID)` assembled together);
//   - the v2 projected list: Array<SessionMessage>  (flat, typed messages)
//
// The Rust normalizer parses only the v1 `{info, parts}` shape, so the TUI
// plugin reassembles the projected list into it at the edge, and `convertV2`
// below still accepts a flat shape for other opencode versions. Everything
// this module does must be a no-op failure: no throws reach the TUI.
//
// Environment:
//   CONTEXT_GUARD_URL          default http://127.0.0.1:7432
//   CONTEXT_GUARD_STATE_DIR    default ~/.local/state/context-guard/opencode
//   CONTEXT_GUARD_CAPTURE_DIR  when set, ship() writes the exact payload to a
//                              file instead of POSTing (fixture capture)

import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

export const SHIP_TIMEOUT_MS = 5_000
export const FETCH_TIMEOUT_MS = 500

export function baseUrl() {
  return process.env.CONTEXT_GUARD_URL ?? "http://127.0.0.1:7432"
}

export function stateDir() {
  const override = process.env.CONTEXT_GUARD_STATE_DIR
  if (override) return override
  return join(homedir(), ".local", "state", "context-guard", "opencode")
}

export function cursorPath(dir = stateDir()) {
  return join(dir, "cursor.json")
}

function readMap(dir) {
  try {
    return JSON.parse(readFileSync(cursorPath(dir), "utf8") ?? "{}")
  } catch {
    return {}
  }
}

export function readCursor(dir = stateDir(), sessionID) {
  const all = readMap(dir)
  return typeof all[sessionID] === "number" ? all[sessionID] : 0
}

export function writeCursor(dir = stateDir(), sessionID, value) {
  try {
    const path = cursorPath(dir)
    const all = { ...readMap(dir), [sessionID]: value }
    mkdirSync(dir, { recursive: true })
    const tmp = `${path}.${process.pid}.${Date.now()}.tmp`
    writeFileSync(tmp, JSON.stringify(all, null, 2))
    renameSync(tmp, path)
  } catch {
    // the cursor is only a reship boundary hint; losing it means a bigger overlap
  }
}

export function isLegacy(message) {
  return !!(message && typeof message === "object" && "info" in message && "parts" in message)
}

function assistantComplete(info) {
  if (info.error) return true
  if (info.time?.completed) return true
  if (info.finish) return true
  return false
}

// ---- conversion of the v2 flat SessionMessage model -----------------------

function convertV2(message) {
  const type = message?.type
  if (type === "assistant") {
    const parts = (message.content ?? [])
      .map((part) => {
        if (part.type === "text") return { type: "text", text: part.text, id: part.id }
        if (part.type === "reasoning") return { type: "reasoning", id: part.id }
        if (part.type !== "tool") return null
        const state = { status: part.state?.status ?? "running", input: part.state?.input ?? {} }
        if (state.status === "completed") state.output = toolOutput(part)
        if (state.status === "error") state.error = part.state?.error?.message ?? ""
        return { type: "tool", id: part.id, callID: part.id, tool: part.name, state }
      })
      .filter(Boolean)
    return {
      info: {
        id: message.id,
        role: "assistant",
        modelID: message.model?.modelID,
        providerID: message.model?.providerID,
        time: message.time,
        tokens: message.tokens,
        error: message.error,
      },
      parts,
    }
  }
  if (type === "user") {
    return {
      info: { id: message.id, role: "user", time: message.time },
      parts: [{ type: "text", text: message.text }],
    }
  }
  if (type === "synthetic") {
    return {
      info: { id: message.id, role: "user", time: message.time },
      parts: [{ type: "text", text: message.text, synthetic: true }],
    }
  }
  // system, compaction, shell and the *-switched events carry no turn
  return null
}

function toolOutput(part) {
  const text = (part?.state?.content ?? [])
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n")
  if (text) return text
  const result = part?.state?.result
  return typeof result === "string" ? result : JSON.stringify(result)
}

// A v2 `shell` message is a tool result that ran between turns; attach it to
// the assistant message that issued the call so the result rides with the
// reply into the next request, where the monitor looks for its call.
function shellToPart(message) {
  if (message?.type !== "shell") return null
  return {
    type: "tool",
    id: message.id,
    callID: message.callID,
    tool: "bash",
    state: { status: "completed", input: { command: message.command ?? "" }, output: message.output ?? "" },
  }
}

// ---- collect: messages since the cursor; a streaming turn holds the tail ----

export function collect(messages, cursor = 0) {
  if (!Array.isArray(messages)) return { records: null, nextCursor: cursor }
  const start = Math.max(0, Math.min(cursor, messages.length))
  const slice = messages.slice(start)

  let lastCommitted = -1
  for (let i = 0; i < slice.length; i++) {
    const message = slice[i]
    if (isLegacy(message)) {
      const info = message.info ?? {}
      if (info.role === "assistant" && !assistantComplete(info)) break
      lastCommitted = i
      continue
    }
    if (shellToPart(message)) {
      lastCommitted = i
      continue
    }
    const converted = convertV2(message)
    if (!converted) continue
    if (converted.info.role === "assistant" && !assistantComplete(converted.info)) break
    lastCommitted = i
  }

  if (lastCommitted < 0) return { records: null, nextCursor: start }

  const records = []
  for (let i = 0; i <= lastCommitted; i++) {
    const message = slice[i]
    if (isLegacy(message)) {
      records.push({ info: message.info, parts: message.parts ?? [] })
      continue
    }
    const shell = shellToPart(message)
    if (shell) {
      const last = records.findLast((r) => r.info.role === "assistant")
      if (last) last.parts.push(shell)
      continue
    }
    const converted = convertV2(message)
    if (converted) records.push(converted)
  }

  // Re-ship the boundary message next time: its tool results must reconstruct
  // the previous reply in the next request, exactly like the Codex hook.
  return { records, nextCursor: start + lastCommitted }
}

export function buildPayload(sessionID, contextLimit, records) {
  return { session_id: sessionID, context_limit: contextLimit ?? null, records }
}

// ---- ship / fetch / status line --------------------------------------------

export async function ship(base = baseUrl(), payload, captureDir) {
  if (captureDir) {
    try {
      const file = join(captureDir, `opencode-${payload.session_id}-${Date.now()}.json`)
      mkdirSync(captureDir, { recursive: true })
      writeFileSync(file, JSON.stringify(payload, null, 2))
      return { accepted: payload.records.length, dropped: 0, captured: file }
    } catch {
      return null
    }
  }
  try {
    const res = await fetch(`${base.replace(/\/$/, "")}/api/v1/ingest/opencode`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(SHIP_TIMEOUT_MS),
    })
    if (res.status !== 202) return null
    return await res.json()
  } catch {
    return null
  }
}

export async function fetchHealth(base = baseUrl(), sessionID) {
  try {
    const res = await fetch(
      `${base.replace(/\/$/, "")}/api/v1/conversations/${encodeURIComponent(sessionID)}/health`,
      { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) },
    )
    if (res.status === 404 || res.status !== 200) return null
    return await res.json()
  } catch {
    return null
  }
}

export function statusLine(health) {
  if (!health) return "Context Guard · waiting for the service"
  const summary = health.summary
  if (typeof summary === "string" && summary.length > 0) return summary
  return `Context Guard ${health.score ?? "?"}`
}

const CONTEXT_START = / · [🟢🟡🟠🔴⚪] context /u

export function splitSummary(text) {
  if (typeof text !== "string" || text.length === 0) return { status: "", context: "" }
  const match = CONTEXT_START.exec(text)
  if (!match) return { status: text, context: "" }
  return {
    status: text.slice(0, match.index),
    context: text.slice(match.index + 3),
  }
}

// Limit resolution: the model the last shipped message used, looked up in the
// TUI's provider registry (`models[modelID].limit.context`).
export function resolveContextLimit(provider, modelID) {
  const models = (provider ?? []).flatMap((p) => Object.entries(p.models ?? {}))
  const entry = models.find(([id]) => id === modelID)
  const context = entry?.[1]?.limit?.context
  return typeof context === "number" && context > 0 ? context : null
}