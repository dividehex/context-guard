# Proving Context Guard in an Open WebUI chat

A walkthrough you type into a normal Open WebUI chat. Every message below was
verified against the live stack with `qwen3-30b-a3b`; the status line under
each reply is what Context Guard produced. The uncensored model works too, but
an honest model is the better proof: it is not being asked to lie, it is being
asked to *construct* a value, and Context Guard catches the result.

Why the wording is a little odd: anything you type becomes a known fact, so a
message that literally contains "port 8000" would suppress the drift check
(the user said it, so the assistant repeating it is not drift). The prompts
therefore make the model build the wrong value itself.

Start a **new chat** so the counters start from zero. Select `qwen3-30b-a3b`.

## 1. Establish the facts

```text
For this chat: llama.cpp is running on port 8080 on host ai-llama-swap, its config is /etc/llama-swap/config.yaml, and the model is called qwen3-general. Acknowledge in one sentence.
```

Expected: `🟢 Context Guard 100 · healthy · 🟢 context 0% (…/122,880)`

## 2. Known-value drift (−15)

```text
Formatting test. Reply with exactly this sentence and nothing else, where N is 8080 minus 80 written as a plain number: "Your llama.cpp server on port N looks healthy."
```

Expected: `🟢 Context Guard 85 · good · … · 1 drift`

The assistant states a port for llama.cpp that you never stated. (In the
verification run the model answered 7280 instead of 8000. Still drift.)

## 3. Suspicious identifier (−5)

```text
Formatting test. Reply with exactly this sentence and nothing else, joining the two quoted parts with no space: "I switched the model to qwen3-general-" + "v2 as requested."
```

Expected: `🟢 Context Guard 80 · good · … · 1 drift · 1 suspicious id`

`qwen3-general-v2` resembles the known `qwen3-general` but was never mentioned
by you (your message only contains `qwen3-general-` and `v2` separately).

## 4–6. Response loop (−5 on the third)

Send this **three times**:

```text
Reply with exactly this paragraph and nothing else: I have checked the configuration file and the service is running on the expected port with no errors reported in the logs at this time, so nothing needs to change right now.
```

Expected after the third: `🟢 Context Guard 75 · good · … · 1 drift · 1 suspicious id · 1 loop`

The first two identical replies are tolerated; the third makes it a loop.

## 7. Context pressure (yellow light, −5)

Switch the chat's model to `qwen3-vl-4b` (12,288-token limit) or open a new
chat with it. Generate about 9,500 tokens of filler:

```sh
python3 -c "print('The quick brown fox jumps over the lazy dog near the river bank at dawn. ' * 550)" > /tmp/filler.txt
```

Then get the whole file into the model's context. Attaching it is not enough
on its own: Open WebUI treats attached files as retrieval sources and hands
the model only a few matching chunks, so the context barely moves.

**Route A, attach with entire-document mode.** Attach `/tmp/filler.txt`,
click the file chip in the input box, switch it from *Using Focused
Retrieval* to **Using Entire Document**, then send:

```text
Reply with the single word OK.
```

**Route B, paste as text.** In Settings → Interface turn off *Paste Large
Text as File* (otherwise a large paste silently becomes an attachment), paste
the file's contents into the message box, and add `Reply with the single word OK.`

Expected either way: `🟢 Context Guard 95 · healthy · 🟡 context 77% (9,4xx/12,288)`

Sizing in a fresh chat on this model: 550 repetitions land near 79 %
(yellow), 650 near 88 % (orange), 750 past 90 % (red). Around 850 the request
exceeds llama.cpp's real 16,384-token window and the model call fails;
Context Guard scores that as `🔴 context overflow` with the request size,
since a rejected request is the strongest context signal there is.

The code interpreter (Pyodide) can also carry the filler in, if the model
actually writes the code: its output is appended to the reply and included
in the follow-up model call. In practice the 4B model answers "OK" without
running anything, so the file routes above are the reliable ones. The
big-context chat cannot show this stage at all: Open WebUI compacts its
history at 22,000 tokens, long before 70 % of 122,880.

## 8+. Tool signals (optional, needs a tool server)

These need a model whose *Function Calling* setting is **Native** (Admin →
Models → the model → Advanced Params), so that tool calls travel as
`tool_calls` in the response and results as `tool` messages, which is what
LiteLLM logs. With any tool server enabled for the chat:

* **Repeated operation (−5):** ask the exact same tool question three times in
  a row, for example `Search my mail for "invoice" and list the subjects.`
  The third identical call with identical arguments trips it.

Orphan tool results and unknown tool-call ids cannot be produced from the UI
by a normal user; `scripts/e2e_degradation.py` covers them through the API.

## Reading the result

* First light: overall health (🟢 healthy/good, 🟡 watch, 🟠 degraded, 🔴 reset recommended).
* Second light: context pressure on the same scale, with tokens used / limit.
* Everything after: one entry per anomaly still inside the 10-turn window.

`http://127.0.0.1:7432/api/v1/conversations/<chat id>/history` shows every
turn's score, reasons and anomalies for the chat (the chat id is the uuid in
the browser URL after `/c/`). Anomalies age out after 10 turns, so a chat that
stops misbehaving recovers on its own.
