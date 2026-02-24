"""Bi-directional worker communication tools.

RequestInputTool — for workers: pause and ask Zero a question.
WorkerReplyTool  — for Zero: send an answer to a paused worker.
"""

import asyncio
from typing import Any, TYPE_CHECKING

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager, WorkerState


class RequestInputTool(Tool):
    """Worker-side tool: pause execution and ask Zero a question.

    When called, the worker's agent loop blocks (via asyncio.Event) until
    Zero replies using worker_reply. This enables multi-turn conversations
    between Zero and her workers.
    """

    # Timeout before the worker gives up waiting and proceeds autonomously.
    INPUT_TIMEOUT = 1800  # 30 minutes

    def __init__(self, worker_state: "WorkerState"):
        self._state = worker_state

    @property
    def name(self) -> str:
        return "request_input"

    @property
    def description(self) -> str:
        return (
            "Ask Zero (your manager) a question when you need clarification, "
            "a decision, or information you don't have access to. "
            "Your execution will pause until she replies. "
            "Include context about what you've found so far."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask Zero",
                },
                "context": {
                    "type": "string",
                    "description": "What you've found/done so far (helps Zero answer better)",
                },
            },
            "required": ["question"],
        }

    async def execute(self, question: str, context: str = "", **kwargs: Any) -> str:
        state = self._state
        wid = state.worker_id

        # Record the question
        state.question = question
        state.question_context = context
        state.status = "waiting_for_input"
        state.input_event.clear()

        logger.info(f"Worker {wid} requesting input: {question[:80]}")

        # Notify Zero via callback (sends MessageBus inbound message)
        if state.notify_callback:
            await state.notify_callback(wid, state.task_id, question, context)

        # Block until Zero replies or timeout
        try:
            await asyncio.wait_for(
                state.input_event.wait(), timeout=self.INPUT_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Worker {wid} timed out waiting for input")
            state.status = "running"
            state.question = None
            return (
                "No answer received within 30 minutes. "
                "Proceed with your best judgment based on what you know."
            )

        answer = state.input_answer or "(no answer provided)"
        state.status = "running"
        state.question = None
        state.input_answer = None

        logger.info(f"Worker {wid} received answer: {answer[:80]}")
        return f"Zero's answer: {answer}"


class WorkerReplyTool(Tool):
    """Zero-side tool: reply to a worker that asked a question.

    Finds the waiting worker and delivers the answer, unblocking
    the worker's request_input call.
    """

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "worker_reply"

    @property
    def description(self) -> str:
        return (
            "Reply to a worker who asked a question via request_input. "
            "The worker is paused and waiting for your answer. "
            "You can also ask Cai first if you're unsure, then reply."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "worker_id": {
                    "type": "string",
                    "description": "The worker/run ID (e.g. W-001) that asked the question",
                },
                "answer": {
                    "type": "string",
                    "description": "Your answer to the worker's question",
                },
            },
            "required": ["worker_id", "answer"],
        }

    async def execute(self, worker_id: str, answer: str, **kwargs: Any) -> str:
        return self._manager.reply_to_worker(worker_id, answer)
