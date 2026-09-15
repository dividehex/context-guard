/** @jsxImportSource @opentui/solid */
import type { TuiPlugin, TuiPluginApi } from "@opencode-ai/plugin/tui"
import type { AssistantMessage, Message, Part } from "@opencode-ai/sdk/v2"
import { createMemo, createSignal, type Accessor, type Setter } from "solid-js"
import os from "node:os"
import path from "node:path"

import { buildPayload, collect, fetchHealth, readCursor, ship, splitSummary, statusLine, writeCursor } from "./context_guard_shipper.mjs"

const DEFAULT_URL = "http://127.0.0.1:7432"

const env = {
  url: process.env.CONTEXT_GUARD_URL ?? DEFAULT_URL,
  stateDir:
    process.env.CONTEXT_GUARD_STATE_DIR ?? path.join(os.homedir(), ".local", "state", "context-guard", "opencode"),
  captureDir: process.env.CONTEXT_GUARD_CAPTURE_DIR,
  notifyBelow: Number(process.env.CONTEXT_GUARD_NOTIFY_BELOW ?? 0) || 0,
}

const POLL_MS = 2000

type Score = {
  text: string
  score: number | null
}

type Entry = {
  id: string
  read: Accessor<Score>
  set: Setter<Score>
  notified: boolean
}

// ---- per-plugin mutable chat state -----------------------------------------

const sessions = new Map<string, Entry>()

function entryFor(sessionID: string): Entry {
  let entry = sessions.get(sessionID)
  if (!entry) {
    const [read, set] = createSignal<Score>({ text: statusLine(null), score: null })
    entry = { id: sessionID, read, set, notified: false }
    sessions.set(sessionID, entry)
  }
  return entry
}

function applyScore(api: TuiPluginApi, sessionID: string, text: string, score: number | null) {
  const entry = entryFor(sessionID)
  entry.set({ text, score })

  if (env.notifyBelow > 0 && score !== null) {
    if (score < env.notifyBelow && !entry.notified) {
      entry.notified = true
      try {
        api.ui.toast({ title: "Context Guard", message: text, variant: score < 60 ? "error" : "warning" })
      } catch {
        // never-fail display
      }
    } else if (score >= env.notifyBelow && entry.notified) {
      entry.notified = false // re-arm on recovery
    }
  }
}

// The SDK's session messages are the `Message` union (role-based), which
// carries no parts: the parts live in the per-message store. Re-attach them
// so the shipper's legacy `{info, parts}` path can ship them, the same shape
// the Rust normalizer and the SDK's own `/session/{id}/message` use.
function messagesOf(api: TuiPluginApi, sessionID: string): Array<{ info: Message; parts: Part[] }> {
  try {
    const messages = api.state.session.messages(sessionID) as unknown as Message[]
    if (!Array.isArray(messages)) return []
    return messages.map((m) => ({ info: m, parts: api.state.part(m.id) as unknown as Part[] }))
  } catch {
    // fall through to the client
  }
  return []
}

function contextLimitFor(api: TuiPluginApi, sessionID: string): number | null {
  const last = messagesOf(api, sessionID)
    .map((record) => record.info)
    .findLast((m): m is AssistantMessage => m.role === "assistant")
  if (!last) return null
  const provider = api.state.provider.find((p) => p.id === last.providerID)
  return provider?.models?.[last.modelID]?.limit?.context ?? null
}

async function shipMessages(api: TuiPluginApi, sessionID: string, messages: Array<{ info: Message; parts: Part[] }>) {
  try {
    const cursor = readCursor(env.stateDir, sessionID)
    const { records, nextCursor } = collect(messages, cursor)
    if (!records || records.length === 0) return

    const payload = buildPayload(sessionID, contextLimitFor(api, sessionID), records)
    const verdict = await ship(env.url, payload, env.captureDir)
    if (!verdict) return // service down; cursor untouched so a bigger overlap fixes it later

    writeCursor(env.stateDir, sessionID, nextCursor)

    const health = await fetchHealth(env.url, sessionID)
    const score = health && typeof health.score === "number" ? health.score : null
    const text =
      health?.summary ?? (score === null ? statusLine(null) : statusLine({ score }))
    applyScore(api, sessionID, text, score)
  } catch {
    // shipping is best-effort; never take the TUI down
  }
}

function flush(api: TuiPluginApi, sessionID: string) {
  if (!sessionID) return

  const messages = messagesOf(api, sessionID)
  if (messages.length > 0) {
    shipMessages(api, sessionID, messages)
    return
  }

  // Reactive store not caught up yet (or server-backed); ask once. The SDK
  // returns the same `{info, parts}` record shape directly.
  api.client.session
    .messages({ sessionID, limit: 1000 })
    .then((res) => {
      const data = (res as { data?: Array<{ info: Message; parts: Part[] }> }).data
      if (Array.isArray(data) && data.length > 0) shipMessages(api, sessionID, data)
    })
    .catch(() => {})
}

// ---------------------------------------------------------------------------

function GuardView(props: { api: TuiPluginApi; sessionID: string }) {
  const entry = entryFor(props.sessionID)
  const theme = () => props.api.theme.current
  const fg = () => {
    const { score } = entry.read() // reactive
    if (score === null) return theme().textMuted
    if (score >= 80) return theme().success
    if (score >= 60) return theme().warning
    return theme().error
  }
  const parts = createMemo(() => splitSummary(entry.read().text))
  const href = () => `${env.url.replace(/\/$/, "")}/ui/conversations/${encodeURIComponent(props.sessionID)}`

  return (
    <box>
      <text fg={theme().text}>
        <b>Context Guard</b>
      </text>
      {parts().context ? (
        <>
          <text fg={fg()}>
            <a href={href()}>{parts().status}</a>
          </text>
          <text fg={fg()}>
            <a href={href()}>{parts().context}</a>
          </text>
        </>
      ) : (
        <text fg={fg()}>{parts().status}</text>
      )}
    </box>
  )
}

const tui: TuiPlugin = async (api) => {
  const flushCurrent = () => {
    const route = api.route.current
    if (route?.name !== "session") return
    const sessionID = (route.params as { sessionID?: string } | undefined)?.sessionID
    if (sessionID) flush(api, sessionID)
  }

  const unsubscribe: Array<() => void> = []
  for (const type of ["session.status", "session.idle", "message.updated"] as const) {
    unsubscribe.push(api.event.on(type, () => flushCurrent()))
  }

  const poll = setInterval(() => {
    try {
      flushCurrent()
    } catch {
      // never-fail
    }
  }, POLL_MS)

  api.lifecycle.onDispose(() => {
    clearInterval(poll)
    unsubscribe.forEach((off) => off())
  })

  api.slots.register({
    order: 150,
    slots: {
      sidebar_content(_ctx, props) {
        return <GuardView api={api} sessionID={props.session_id} />
      },
    },
  })
}

const plugin = {
  id: "context-guard",
  tui,
}

export default plugin