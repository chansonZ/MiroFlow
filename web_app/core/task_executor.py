# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""Background task execution for agent runs."""

import asyncio
import json
import logging
import os
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from ..core.config import AppConfig
from ..core.session_manager import SessionManager
from ..models.task import FileInfo

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes agent tasks in background threads."""

    def __init__(self, config: AppConfig, session_manager: SessionManager):
        self.config = config
        self.session_manager = session_manager
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self._running_tasks: dict[str, threading.Thread] = {}
        self._task_tracers: dict[str, Any] = {}

    def submit_task(
        self,
        task_id: str,
        task_description: str,
        config_path: str,
        file_info: FileInfo | None = None,
    ) -> None:
        """Submit a task for background execution."""
        thread = threading.Thread(
            target=self._run_task_sync,
            args=(task_id, task_description, config_path, file_info),
            daemon=True,
        )
        self._running_tasks[task_id] = thread
        thread.start()

    def _run_task_sync(
        self,
        task_id: str,
        task_description: str,
        config_path: str,
        file_info: FileInfo | None,
    ) -> None:
        """Synchronous wrapper for async task execution."""
        asyncio.run(self._run_task(task_id, task_description, config_path, file_info))

    async def _run_task(
        self,
        task_id: str,
        task_description: str,
        config_path: str,
        file_info: FileInfo | None,
    ) -> None:
        """Execute agent task asynchronously."""
        # Change to project root for relative imports
        os.chdir(self.config.project_root)

        tracer = None

        try:
            # Import MiroFlow components (import here to avoid circular imports)
            from config import load_config
            from miroflow.agents import build_agent_from_config
            from miroflow.agents.context import AgentContext
            from miroflow.logging.task_tracer import get_tracer, set_tracer

            # Update status to running
            self.session_manager.update_task(task_id, {"status": "running"})

            # Create unique output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = str(uuid.uuid4())[:8]
            output_dir = self.config.logs_dir / f"{timestamp}_{run_id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load configuration
            cfg = load_config(config_path, f"output_dir={output_dir}")

            # Get max_turns from config
            max_turns = 30
            if hasattr(cfg, "main_agent") and hasattr(cfg.main_agent, "max_turns"):
                max_turns = cfg.main_agent.max_turns

            # Update session with log path and max_turns
            self.session_manager.update_task(
                task_id,
                {
                    "log_path": str(output_dir),
                    "max_turns": max_turns,
                },
            )

            # Setup tracer
            set_tracer(cfg.output_dir)
            tracer = get_tracer()
            tracer.set_log_path(cfg.output_dir)
            self._task_tracers[task_id] = tracer

            # Build agent
            agent = build_agent_from_config(cfg=cfg)

            # Build context
            ctx_kwargs: dict[str, Any] = {"task_description": task_description}

            if file_info:
                # Pass the absolute file path as task_file_name (string)
                # This is what InputMessageGenerator expects
                ctx_kwargs["task_file_name"] = file_info.absolute_file_path

            ctx = AgentContext(**ctx_kwargs)

            # Start tracer
            tracer.start()
            tracer.update_task_meta(
                {
                    "task_id": task_id,
                    "task_description": task_description,
                }
            )

            # Run agent
            result = await agent.run(ctx)

            # Get final message history and trajectory before cleanup
            final_data = self._get_all_messages_from_tracer(tracer)

            # Update session with results and full message history
            self.session_manager.update_task(
                task_id,
                {
                    "status": "completed",
                    "final_answer": result.get("final_boxed_answer", ""),
                    "summary": result.get("summary", ""),
                    "messages": final_data["messages"],
                    "trajectory": final_data["trajectory"],
                },
            )

            tracer.finish(status="completed")

        except Exception as e:
            error_msg = f"{e!s}\n{traceback.format_exc()}"
            self.session_manager.update_task(
                task_id,
                {
                    "status": "failed",
                    "error_message": error_msg,
                },
            )
            if tracer:
                tracer.finish(status="failed", error=str(e))

        finally:
            # Cleanup
            if task_id in self._running_tasks:
                del self._running_tasks[task_id]
            if task_id in self._task_tracers:
                del self._task_tracers[task_id]

    def _get_all_messages_from_tracer(self, tracer: Any) -> dict[str, Any]:
        """Extract all messages and trajectory from tracer for persistence."""
        try:
            with tracer._data_lock:
                for key, log_file in tracer._active_tasks.items():
                    agent_states = log_file.agent_states
                    for agent_name, state in agent_states.items():
                        state_data = (
                            state.state
                            if hasattr(state, "state")
                            else state.get("state", {})
                        )
                        message_history = state_data.get("message_history", [])
                        return {
                            "messages": self._format_messages(message_history),
                            "trajectory": self._build_trajectory(message_history),
                        }
        except Exception:
            logger.debug("Failed to retrieve task messages", exc_info=True)
        return {"messages": [], "trajectory": []}

    def get_task_progress(self, task_id: str) -> dict[str, Any]:
        """Get current progress from tracer."""
        tracer = self._task_tracers.get(task_id)
        if tracer is None:
            return {
                "current_turn": 0,
                "step_count": 0,
                "recent_logs": [],
                "messages": [],
                "trajectory": [],
            }

        try:
            with tracer._data_lock:
                for key, log_file in tracer._active_tasks.items():
                    agent_states = log_file.agent_states
                    step_logs = log_file.step_logs

                    # Calculate turn count and get message history
                    current_turn = 0
                    messages = []
                    for agent_name, state in agent_states.items():
                        state_data = (
                            state.state
                            if hasattr(state, "state")
                            else state.get("state", {})
                        )
                        message_history = state_data.get("message_history", [])
                        current_turn = max(
                            current_turn, (len(message_history) + 1) // 2
                        )
                        # Get ALL messages for display (full history)
                        messages = self._format_messages(message_history)

                    # Filter and format logs to show tool calls
                    recent_logs = (
                        self._format_recent_logs(step_logs[-30:]) if step_logs else []
                    )

                    # Build structured trajectory from message history
                    trajectory = self._build_trajectory(message_history)

                    return {
                        "current_turn": current_turn,
                        "step_count": len(step_logs),
                        "recent_logs": recent_logs,
                        "messages": messages,
                        "trajectory": trajectory,
                    }
        except Exception:
            logger.debug("Failed to retrieve task progress", exc_info=True)

        return {"current_turn": 0, "step_count": 0, "recent_logs": [], "messages": [], "trajectory": []}

    def _format_recent_logs(self, logs: list[dict]) -> list[dict]:
        """Format and filter logs to show relevant tool call and LLM information."""
        formatted = []
        for log in logs:
            log_type = log.get("type", "")

            # Include tool calls and results
            if (
                "tool" in log_type.lower()
                or log.get("tool_name")
                or log.get("server_name")
            ):
                formatted.append(
                    {
                        "type": "tool_call",
                        "tool_name": log.get("tool_name", ""),
                        "server_name": log.get("server_name", ""),
                        "input": self._truncate_output(
                            log.get("input") or log.get("arguments") or log.get("args")
                        ),
                        "output": self._truncate_output(
                            log.get("output") or log.get("result") or log.get("content")
                        ),
                    }
                )
            # Include LLM calls
            elif "llm" in log_type.lower() or log.get("model") or log.get("prompt"):
                formatted.append(
                    {
                        "type": "llm_call",
                        "model": log.get("model", ""),
                        "input": self._truncate_output(
                            log.get("prompt") or log.get("input") or log.get("messages")
                        ),
                        "output": self._truncate_output(
                            log.get("response")
                            or log.get("output")
                            or log.get("content")
                        ),
                    }
                )
            # Include span events (shows execution flow)
            elif log_type in ("span_start", "span_end"):
                path = log.get("path", "")
                # Only include interesting spans
                if any(
                    x in path.lower()
                    for x in ["tool", "search", "read", "llm", "agent", "mcp"]
                ):
                    formatted.append(
                        {
                            "type": log_type,
                            "path": path,
                            "name": path.split("/")[-1] if path else "",
                        }
                    )
            # Include any log with tool-related or LLM-related fields
            elif any(
                key in log
                for key in ["tool_name", "server_name", "tool_call", "model", "prompt"]
            ):
                formatted.append(log)

        # Return last 15 formatted logs
        return formatted[-15:]

    def _format_messages(self, messages: list[dict]) -> list[dict]:
        """Format message history for display.

        Note: We do NOT truncate message content here to preserve:
        - Full thinking/reasoning content for proper display
        - Complete tool results (e.g., search results JSON) for parsing
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle different content types
            if isinstance(content, list):
                # Tool results or multi-part content
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "tool_result":
                            # Don't truncate tool results - need full JSON for parsing
                            tool_content = item.get("content", "")
                            if isinstance(tool_content, str):
                                text_parts.append(tool_content)
                            else:
                                text_parts.append(str(tool_content))
                        elif item.get("type") == "tool_use":
                            text_parts.append(f"[Tool Call: {item.get('name', '')}]")
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "\n".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)

            # Don't truncate - preserve full content for thinking and tool results
            formatted_msg: dict = {
                "role": role,
                "content": content,
            }

            # For assistant messages, also include dedicated reasoning as <think> block
            # so the frontend fallback (parseMessageContent) can render it.
            if role == "assistant":
                dedicated = msg.get("reasoning") or msg.get("reasoning_content")
                if dedicated and isinstance(dedicated, str) and dedicated.strip():
                    formatted_msg["content"] = (
                        f"<think>{dedicated.strip()}</think>\n{content}"
                    )

            formatted.append(formatted_msg)

        return formatted

    def _truncate_output(self, output: Any) -> Any:
        """Truncate long output strings."""
        if output is None:
            return None
        if isinstance(output, str) and len(output) > 1000:
            return output[:1000] + "... (truncated)"
        return output

    # ------------------------------------------------------------------
    # Trajectory building
    # ------------------------------------------------------------------

    def _build_trajectory(self, message_history: list[dict]) -> list[dict]:
        """Build a structured trajectory from the raw message_history.

        Supports both:
        - Native OpenAI tool_calls (role=assistant with tool_calls list, role=tool)
        - MCP XML format (<use_mcp_tool>...</use_mcp_tool> / <tool_result>)
        """
        events: list[dict] = []
        counter = [0]

        def new_id(prefix: str = "evt") -> str:
            counter[0] += 1
            return f"{prefix}_{counter[0]}"

        # Pending tool calls awaiting results
        native_pending: dict[str, dict] = {}  # call_id -> event
        mcp_pending: list[dict] = []  # ordered, matches user messages in sequence

        # Context for assigning parent_id to reasoning events
        last_completed_type: str | None = None  # "search" | "read"
        last_search_id: str | None = None
        last_read_id: str | None = None

        for msg in message_history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                # Determine reasoning parent from last completed tool
                if last_completed_type == "search":
                    reasoning_parent = last_search_id
                elif last_completed_type == "read":
                    reasoning_parent = last_read_id
                else:
                    reasoning_parent = None

                # Extract <think> block from content string
                think_extracted = False
                if isinstance(content, str):
                    think_match = re.search(
                        r"<think>([\s\S]*?)</think>", content, re.IGNORECASE
                    )
                    if think_match:
                        reasoning_text = think_match.group(1).strip()
                        if reasoning_text:
                            evt_id = new_id("r")
                            events.append(
                                {
                                    "id": evt_id,
                                    "type": "reasoning",
                                    "text": reasoning_text,
                                    "parent_id": reasoning_parent,
                                }
                            )
                            think_extracted = True

                # Also check dedicated reasoning field (e.g., Kimi/DeepSeek via OpenRouter)
                # saved by process_llm_response as assistant_message["reasoning"]
                if not think_extracted:
                    dedicated_reasoning = msg.get("reasoning") or msg.get(
                        "reasoning_content"
                    )
                    if (
                        dedicated_reasoning
                        and isinstance(dedicated_reasoning, str)
                        and dedicated_reasoning.strip()
                    ):
                        evt_id = new_id("r")
                        events.append(
                            {
                                "id": evt_id,
                                "type": "reasoning",
                                "text": dedicated_reasoning.strip(),
                                "parent_id": reasoning_parent,
                            }
                        )

                # Handle native OpenAI tool_calls
                native_tool_calls = msg.get("tool_calls", [])
                if native_tool_calls and isinstance(native_tool_calls, list):
                    for tc in native_tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        call_id = tc.get("id", new_id("call"))
                        fn = tc.get("function", {})
                        tool_name = fn.get("name", "") if isinstance(fn, dict) else str(fn)
                        args_str = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else {}
                        except Exception:
                            args = {"raw": str(args_str)[:500]}

                        evt_id = new_id("t")
                        evt = self._make_tool_event(evt_id, tool_name, args, last_search_id)
                        events.append(evt)
                        native_pending[call_id] = evt
                        if evt["type"] == "search":
                            last_search_id = evt_id

                # Handle MCP XML tool calls
                if isinstance(content, str) and "<use_mcp_tool" in content:
                    mcp_pattern = re.compile(
                        r"<use_mcp_tool[^>]*>.*?"
                        r"(?:<server_name[^>]*>(.*?)</server_name>.*?)?"
                        r"<tool_name[^>]*>(.*?)</tool_name>.*?"
                        r"<arguments[^>]*>([\s\S]*?)</arguments>.*?"
                        r"</use_mcp_tool>",
                        re.IGNORECASE | re.DOTALL,
                    )
                    for match in mcp_pattern.finditer(content):
                        tool_name = (match.group(2) or "").strip()
                        args_str = (match.group(3) or "").strip()
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except Exception:
                            args = {"raw": args_str[:500]}

                        evt_id = new_id("t")
                        evt = self._make_tool_event(evt_id, tool_name, args, last_search_id)
                        events.append(evt)
                        mcp_pending.append(evt)
                        if evt["type"] == "search":
                            last_search_id = evt_id

            elif role == "tool":
                # Native tool result message
                call_id = msg.get("tool_call_id", "")
                raw_content = content
                if isinstance(raw_content, list):
                    parts = []
                    for item in raw_content:
                        if isinstance(item, dict):
                            parts.append(
                                item.get("text") or item.get("content") or ""
                            )
                        else:
                            parts.append(str(item))
                    raw_content = "\n".join(parts)
                elif not isinstance(raw_content, str):
                    raw_content = str(raw_content)

                matched_evt = native_pending.get(call_id)
                if matched_evt:
                    self._apply_result_to_event(matched_evt, raw_content)
                    if matched_evt["type"] == "search":
                        last_completed_type = "search"
                        last_search_id = matched_evt["id"]
                    elif matched_evt["type"] == "read":
                        last_completed_type = "read"
                        last_read_id = matched_evt["id"]

            elif role == "user":
                # Collect text from user message (tool results in MCP / text protocol)
                texts: list[str] = []
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            t = item.get("text") or item.get("content") or ""
                            if t:
                                texts.append(str(t))
                        elif isinstance(item, str):
                            texts.append(item)

                full_text = "\n".join(texts)
                if not full_text:
                    continue

                # Try to match <tool_result>name:\ncontent</tool_result> blocks
                tool_result_re = re.compile(
                    r"<tool_result>\s*(\S[^:\n]*?):\s*([\s\S]*?)</tool_result>",
                    re.IGNORECASE,
                )
                tool_result_matches = list(tool_result_re.finditer(full_text))

                if tool_result_matches:
                    for rm in tool_result_matches:
                        tool_name_in_result = rm.group(1).strip()
                        result_text = rm.group(2).strip()

                        # Match to first unresolved MCP pending call with same tool_name
                        matched = None
                        for evt in mcp_pending:
                            if (
                                evt.get("tool_name") == tool_name_in_result
                                and not evt.get("_resolved")
                            ):
                                matched = evt
                                break
                        if not matched:
                            # Fallback: any first unresolved
                            for evt in mcp_pending:
                                if not evt.get("_resolved"):
                                    matched = evt
                                    break

                        if matched:
                            matched["_resolved"] = True
                            self._apply_result_to_event(matched, result_text)
                            if matched["type"] == "search":
                                last_completed_type = "search"
                                last_search_id = matched["id"]
                            elif matched["type"] == "read":
                                last_completed_type = "read"
                                last_read_id = matched["id"]

                elif mcp_pending:
                    # text_protocol: whole content is the tool result for the first pending call
                    for evt in mcp_pending:
                        if not evt.get("_resolved"):
                            evt["_resolved"] = True
                            self._apply_result_to_event(evt, full_text)
                            if evt["type"] == "search":
                                last_completed_type = "search"
                                last_search_id = evt["id"]
                            elif evt["type"] == "read":
                                last_completed_type = "read"
                                last_read_id = evt["id"]
                            break

        # Remove internal tracking keys before returning
        for evt in events:
            evt.pop("_resolved", None)

        return events

    def _make_tool_event(
        self,
        evt_id: str,
        tool_name: str,
        args: dict,
        last_search_id: str | None,
    ) -> dict:
        """Create a trajectory event dict for a tool call."""
        tool_lower = tool_name.lower()

        if any(
            x in tool_lower
            for x in ("search", "google", "serper", "bing", "ddg", "tavily", "serpapi")
        ):
            query = str(
                args.get("query")
                or args.get("q")
                or args.get("search_query")
                or args.get("keyword")
                or ""
            )
            return {
                "id": evt_id,
                "type": "search",
                "query": query,
                "results": [],
                "results_count": 0,
                "parent_id": None,
                "tool_name": tool_name,
                "args": args,
            }

        if any(
            x in tool_lower
            for x in ("scrape", "read", "fetch", "browse", "webpage", "url", "crawl", "visit")
        ):
            url = str(
                args.get("url")
                or args.get("webpage_url")
                or args.get("link")
                or args.get("uri")
                or ""
            )
            return {
                "id": evt_id,
                "type": "read",
                "url": url,
                "parent_id": last_search_id,
                "tool_name": tool_name,
                "args": args,
            }

        return {
            "id": evt_id,
            "type": "tool_call",
            "tool_name": tool_name,
            "args": args,
            "parent_id": last_search_id,
        }

    def _apply_result_to_event(self, evt: dict, result_str: str) -> None:
        """Enrich a trajectory event with tool result data (in-place)."""
        if evt.get("type") == "search":
            results = self._parse_search_results(result_str)
            if results:
                evt["results"] = results
                evt["results_count"] = len(results)

    def _parse_search_results(self, result_str: str) -> list[dict]:
        """Parse a JSON search-result string into a list of result dicts."""
        if not result_str:
            return []
        try:
            data = json.loads(result_str)
            organic: list | None = None
            if isinstance(data, dict):
                organic = (
                    data.get("organic")
                    or data.get("organic_results")
                    or data.get("results")
                )
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                if data[0].get("link") or data[0].get("url"):
                    organic = data

            if organic and isinstance(organic, list):
                return [
                    {
                        "title": r.get("title"),
                        "url": r.get("link") or r.get("url") or "",
                        "snippet": r.get("snippet") or r.get("description"),
                        "favicon": r.get("favicon"),
                    }
                    for r in organic[:10]
                    if r.get("link") or r.get("url")
                ]
        except Exception:
            pass
        return []

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task (best effort - marks as cancelled)."""
        if task_id in self._running_tasks:
            self.session_manager.update_task(
                task_id,
                {
                    "status": "cancelled",
                    "error_message": "Task cancelled by user",
                },
            )
            return True
        return False

    def is_task_running(self, task_id: str) -> bool:
        """Check if a task is currently running."""
        return task_id in self._running_tasks
