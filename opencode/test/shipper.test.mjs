import { createServer } from "node:http"
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"
import assert from "node:assert/strict"

import {
  buildPayload,
  collect,
  fetchHealth,
  readCursor,
  resolveContextLimit,
  ship,
  splitSummary,
  statusLine,
  writeCursor,
} from "../context_guard_shipper.mjs"

const SID = "opc_sess"
const T = () => ({ created: 1_755_000_000_000 })
const COMPLETED = () => ({ created: 1_755_000_000_000, completed: 1_755_000_000_500 })

function v2User(id, text) {
  return { id, type: "user", time: T(), text }
}

function v2Synthetic(id, text) {
  return { id, type: "synthetic", time: T(), text }
}

function v2System(id, text) {
  return { id, type: "system", time: T(), text }
}

function v2Compaction(id, summary) {
  return { id, type: "compaction", reason: "manual", summary, recent: "", time: T() }
}

function v2Shell(id, callID, command, output) {
  return { id, type: "shell", callID, command, output, time: { ...T(), completed: 1_755_000_000_400 } }
}

function v2Assistant(id, { text, tool, incomplete = false, error } = {}) {
  const content = []
  if (text) content.push({ type: "text", id: `${id}-t`, text })
  if (tool) content.push(tool)
  return {
    id,
    type: "assistant",
    agent: "primary",
    model: { providerID: "anthropic", modelID: "claude-sonnet-4-5" },
    content,
    tokens: { input: 1000, output: 20, reasoning: 0, cache: { read: 0, write: 0 } },
    time: incomplete ? T() : COMPLETED(),
    ...(error ? { error: { type: "unknown", message: error } } : {}),
  }
}

function v2Tool(callID, name, status, { text, error } = {}) {
  return {
    type: "tool",
    id: callID,
    name,
    state: {
      status,
      input: { command: "cat notes.txt" },
      content: typeof text === "string" ? [{ type: "text", text }] : [],
    },
    ...(status === "error" ? { result: { type: "error", message: error } } : {}),
  }
}

test("cursor files round-trip and write atomically", () => {
  const dir = mkdtempSync(join(tmpdir(), "cg-cursor-"))
  try {
    assert.equal(readCursor(dir, SID), 0)
    writeCursor(dir, SID, 4)
    assert.equal(readCursor(dir, SID), 4)
    writeCursor(dir, "other", 1)
    assert.equal(readCursor(dir, SID), 4, "session cursors are keyed separately")
    assert.equal(readdirSync(dir).length, 1, "only cursor.json remains after atomic rename")
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test("collect converts a v2 session to the {info, parts} ingest body", () => {
  const messages = [
    v2User("u1", "Read notes.txt and tell me the port"),
    v2Assistant("m1", {
      text: "On it.",
      tool: v2Tool("call_1", "bash", "completed", { text: "llama.cpp is running on port 8080\n" }),
    }),
    v2System("sys1", "work in /w"),
    v2Shell("sh1", "call_1", "cat notes.txt", "llama.cpp is running on port 8080\n"),
    v2Compaction("comp1", "the port was 8080"),
    v2Synthetic("syn1", "<system>work in /w</system>"),
    v2User("u2", "what port?"),
    v2Assistant("m2", { text: "port 8000." }),
  ]
  const { records, nextCursor } = collect(messages)

  assert.equal(records.length, 5, "system and compaction are skipped, the shell folds into m1")
  assert.equal(nextCursor, 7)

  assert.deepEqual(records[0], {
    info: { id: "u1", role: "user", time: T() },
    parts: [{ type: "text", text: "Read notes.txt and tell me the port" }],
  })
  assert.equal(records[1].info.role, "assistant")
  assert.equal(records[1].info.modelID, "claude-sonnet-4-5")
  assert.equal(records[1].parts[0].type, "text")
  assert.equal(records[1].parts[1].type, "tool")
  assert.equal(records[1].parts[1].callID, "call_1")
  assert.equal(records[1].parts[1].state.output, "llama.cpp is running on port 8080\n")
  assert.equal(records[2].info.role, "user")
  assert.equal(records[2].parts[0].synthetic, true, "synthetic text keeps its marker")
  assert.equal(records[3].info.id, "u2")

  // The next slice re-ships the boundary message so its results reconstruct
  // the previous reply in the next request.
  const overlap = collect(messages, nextCursor)
  assert.equal(overlap.records[0].info.id, "m2")
  assert.equal(overlap.records[0].parts[0].text, "port 8000.")
})

test("a streaming assistant message holds the tail back", () => {
  const messages = [
    v2User("u1", "hello"),
    v2Assistant("m1", { text: "hi" }),
    v2User("u2", "again"),
    v2Assistant("m2", { text: "thinking…", incomplete: true }),
  ]
  const { records, nextCursor } = collect(messages)
  assert.equal(records.length, 3, "the in-flight reply is not shipped")
  assert.equal(nextCursor, 2, "cursor stops before the incomplete message")
  assert.equal(messages.length, 4)
})

test("collect passes legacy {info, parts} records through", () => {
  const legacy = [
    { info: { id: "u1", role: "user" }, parts: [{ type: "text", text: "hi" }] },
    { info: { id: "m1", role: "assistant", time: COMPLETED() }, parts: [{ type: "text", text: "yo" }] },
  ]
  const { records, nextCursor } = collect(legacy)
  assert.deepEqual(records, legacy)
  assert.equal(nextCursor, 1)
})

test("buildPayload wraps records in the ingest envelope", () => {
  const payload = buildPayload(SID, 200_000, [{ info: { id: "m1" } }])
  assert.deepEqual(payload, { session_id: SID, context_limit: 200_000, records: [{ info: { id: "m1" } }] })
  assert.equal(buildPayload(SID, null, []).context_limit, null)
})

test("ship posts the body and returns the service verdict", async () => {
  const seen = []
  const server = createServer((req, res) => {
    seen.push(req.url)
    res.writeHead(202, { "content-type": "application/json" })
    res.end(JSON.stringify({ accepted: 1, dropped: 0 }))
  })
  await new Promise((resolve) => server.listen(0, resolve))
  try {
    const base = `http://127.0.0.1:${server.address().port}`
    const payload = buildPayload(SID, null, [{ info: { id: "m1" } }])
    const result = await ship(base, payload)
    assert.deepEqual(result, { accepted: 1, dropped: 0 })
    assert.equal(seen.length, 1)
    assert.match(seen[0], /\/api\/v1\/ingest\/opencode$/)
  } finally {
    server.close()
  }
})

test("ship never throws: refused connection and bad status return null", async () => {
  const server = createServer((req, res) => {
    res.writeHead(500)
    res.end("boom")
  })
  await new Promise((resolve) => server.listen(0, resolve))
  try {
    const base = `http://127.0.0.1:${server.address().port}`
    const payload = buildPayload(SID, null, [])
    assert.equal(await ship(base, payload), null)
  } finally {
    server.close()
  }
  assert.equal(await ship("http://127.0.0.1:1", buildPayload(SID, null, [])), null)
})

test("ship with capture dir writes the exact payload to a file", async () => {
  const dir = mkdtempSync(join(tmpdir(), "cg-capture-"))
  try {
    const payload = buildPayload(SID, 8192, [{ info: { id: "m1" } }])
    const result = await ship("http://127.0.0.1:1", payload, dir)
    assert.equal(result.accepted, 1)
    const [file] = readdirSync(dir)
    assert.match(file, /^opencode-opc_sess-\d+\.json$/)
    assert.deepEqual(JSON.parse(readFileSync(join(dir, file), "utf8")), payload)
    assert.equal(
      (await ship("http://127.0.0.1:1", payload, dir)).accepted,
      1,
      "capture mode ignores the base url and still never throws",
    )
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test("fetchHealth returns the document or null", async () => {
  const server = createServer((req, res) => {
    res.writeHead(200, { "content-type": "application/json" })
    res.end(JSON.stringify({ summary: "🟢 Context Guard 100", score: 100 }))
  })
  await new Promise((resolve) => server.listen(0, resolve))
  try {
    const base = `http://127.0.0.1:${server.address().port}`
    assert.deepEqual(await fetchHealth(base, SID), { summary: "🟢 Context Guard 100", score: 100 })
  } finally {
    server.close()
  }
  assert.equal(await fetchHealth("http://127.0.0.1:1", SID), null)
})

test("statusLine prefers the server summary and never throws", () => {
  assert.equal(statusLine(null), "Context Guard · waiting for the service")
  assert.equal(statusLine({ summary: "🟡 Context Guard 74 · watch" }), "🟡 Context Guard 74 · watch")
  assert.equal(statusLine({ score: 100 }), "Context Guard 100")
})

test("splitSummary splits the one-line summary at the context phrase", () => {
  assert.deepEqual(splitSummary("🟢 Context Guard 100 · healthy · 🟢 context 6% (12,586/200,000)"), {
    status: "🟢 Context Guard 100 · healthy",
    context: "🟢 context 6% (12,586/200,000)",
  })
  assert.deepEqual(
    splitSummary("🟡 Context Guard 75 · good · 🟡 context 78% (7,820/10,000) · 1 drift · 1 repeated call"),
    {
      status: "🟡 Context Guard 75 · good",
      context: "🟡 context 78% (7,820/10,000) · 1 drift · 1 repeated call",
    },
  )
  assert.deepEqual(splitSummary("🟢 Context Guard 80 · good · 🔴 context overflow (16,456/12,288)"), {
    status: "🟢 Context Guard 80 · good",
    context: "🔴 context overflow (16,456/12,288)",
  })
  assert.deepEqual(splitSummary("🟢 Context Guard 92 · healthy · ⚪ context ? (508 tokens, limit unknown)"), {
    status: "🟢 Context Guard 92 · healthy",
    context: "⚪ context ? (508 tokens, limit unknown)",
  })
})

test("splitSummary passes through text without a context segment", () => {
  assert.deepEqual(splitSummary("Context Guard · waiting for the service"), {
    status: "Context Guard · waiting for the service",
    context: "",
  })
  assert.deepEqual(splitSummary(""), { status: "", context: "" })
  assert.deepEqual(splitSummary(null), { status: "", context: "" })
})

test("resolveContextLimit looks the model up in the provider registry", () => {
  const provider = [
    { id: "anthropic", models: { "claude-sonnet-4-5": { limit: { context: 200_000 } } } },
  ]
  assert.equal(resolveContextLimit(provider, "claude-sonnet-4-5"), 200_000)
  assert.equal(resolveContextLimit(provider, "missing"), null)
  assert.equal(resolveContextLimit(undefined, "x"), null)
})