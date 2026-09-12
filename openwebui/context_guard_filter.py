"""
title: Context Guard
author: context-guard
version: 0.1.0
description: Shows the Context Guard conversation-health score under each reply as a UI-only status line. Never modifies messages, prompts, or responses.
required_open_webui_version: 0.6.0
"""

# Open WebUI Filter function for Context Guard.
#
# Runs in the outlet only, after the reply has been delivered and persisted.
# It reads the health result for this exact reply from Context Guard and emits
# a `status` event, which Open WebUI stores in the message's statusHistory and
# renders under the reply. Status history is never part of what Open WebUI
# sends to the model, so the score costs zero context tokens.
#
# Fault isolation: every network call has a short timeout, every error is
# swallowed, and the body is always returned unchanged. If Context Guard is
# down this outlet costs at most `connect_timeout` seconds per reply.

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

log = logging.getLogger("context_guard_filter")

STATUS_ORDER = ["healthy", "good", "watch", "degraded", "reset_recommended"]


class Filter:
    class Valves(BaseModel):
        context_guard_url: str = Field(
            default="http://context-guard:7432",
            description="Base URL of the Context Guard service, reachable from the Open WebUI container.",
        )
        wait_seconds: float = Field(
            default=6.0,
            description="Total time to wait for the score before giving up silently.",
        )
        poll_interval: float = Field(default=0.5, description="Seconds between polls while the score is not ready.")
        connect_timeout: float = Field(default=1.0, description="Per-request timeout in seconds.")
        show_minimum: str = Field(
            default="always",
            description="Show the status line for every reply ('always') or only from this status downward: good, watch, degraded, reset_recommended.",
        )
        notify_below: int = Field(
            default=40,
            description="Also raise a toast notification when health drops below this value; 0 disables.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[Any]]] = None,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        try:
            await self._report(body, __event_emitter__, __metadata__ or {})
        except Exception as e:  # never let monitoring break a chat
            log.debug("context guard filter skipped: %s", e)
        return body

    async def _report(self, body: dict, emitter, metadata: dict) -> None:
        if emitter is None:
            return
        chat_id = metadata.get("chat_id") or body.get("chat_id")
        message_id = metadata.get("message_id") or body.get("id")
        if not chat_id or not message_id:
            return

        result = await self._wait_for_result(chat_id, message_id, self._reply_timestamp(body))
        if result is None:
            return

        summary = result.get("summary") or f"Context Guard {result.get('score')}"
        status = str(result.get("status") or "")
        if self._should_show(status):
            await emitter({"type": "status", "data": {"description": summary, "done": True}})

        score = result.get("score")
        if isinstance(score, (int, float)) and 0 < self.valves.notify_below and score < self.valves.notify_below:
            await emitter(
                {
                    "type": "notification",
                    "data": {"type": "warning", "content": f"Context Guard: this conversation scored {int(score)}. Consider starting a new chat."},
                }
            )

    def _should_show(self, status: str) -> bool:
        minimum = (self.valves.show_minimum or "always").strip().lower()
        if minimum == "always" or minimum not in STATUS_ORDER:
            return True
        if status not in STATUS_ORDER:
            return True
        return STATUS_ORDER.index(status) >= STATUS_ORDER.index(minimum)

    @staticmethod
    def _reply_timestamp(body: dict) -> Optional[float]:
        """Unix time of the last assistant message, for the fallback lookup."""
        for m in reversed(body.get("messages") or []):
            if m.get("role") == "assistant":
                ts = m.get("timestamp")
                if isinstance(ts, (int, float)) and ts > 0:
                    return float(ts)
                return None
        return None

    async def _wait_for_result(self, chat_id: str, message_id: str, reply_ts: Optional[float]) -> Optional[dict]:
        base = self.valves.context_guard_url.rstrip("/")
        by_message = f"{base}/api/v1/conversations/{chat_id}/health?message_id={message_id}"
        by_time = f"{base}/api/v1/conversations/{chat_id}/health?after={reply_ts - 1:.3f}" if reply_ts else None
        deadline = time.monotonic() + max(0.0, self.valves.wait_seconds)
        timeout = aiohttp.ClientTimeout(total=max(0.1, self.valves.connect_timeout))
        used_fallback = False

        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                status, payload = await self._get(session, by_message)
                if status == 200:
                    return payload
                if status == 404:
                    # Not scored yet (or, for a brand-new chat, not even known yet: LiteLLM flushes
                    # its batch up to a second after the reply). Keep polling. After the deadline,
                    # try the timestamp fallback once in case the message-id header is not configured.
                    if time.monotonic() >= deadline:
                        if by_time and not used_fallback:
                            used_fallback = True
                            s, p = await self._get(session, by_time)
                            return p if s == 200 else None
                        return None
                    await asyncio.sleep(max(0.05, self.valves.poll_interval))
                    continue
                return None  # Context Guard unreachable or erroring: give up quietly

    @staticmethod
    async def _get(session: aiohttp.ClientSession, url: str):
        try:
            async with session.get(url) as resp:
                try:
                    payload = await resp.json(content_type=None)
                except Exception:
                    payload = {}
                return resp.status, payload if isinstance(payload, dict) else {}
        except Exception as e:
            log.debug("context guard request failed: %s", e)
            return None, {}
