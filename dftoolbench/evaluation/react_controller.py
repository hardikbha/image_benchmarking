"""
react_controller.py — ReAct agent controller for DFToolBench-I.

Implements the Reasoning + Acting (ReAct) prompting framework for
tool-augmented deepfake forensic agents.

ReAct protocol loop (per turn)
------------------------------
1. **Thought**       — model reasons about the current state.
2. **Action**        — model names the tool to invoke.
3. **Action Input**  — model provides the tool arguments (JSON or plain text).
4. **Observation**   — the controller executes the tool and appends the result.

The loop continues until the model emits a ``Final Answer`` action or the
maximum number of turns is reached.

Operation modes
---------------
``agent``
    Full ReAct loop: the controller calls tools and feeds observations back.
``e2e``
    End-to-end direct-answer mode: a single forward pass produces the
    final answer without tool invocations.

Configuration via environment variables
----------------------------------------
ANTHROPIC_API_KEY
    Required when using a Claude model.
GEMINI_API_KEY
    Required when using a Gemini model.
DFTOOLBENCH_MODEL
    Default model name (overridden by constructor argument).
DFTOOLBENCH_MAX_TURNS
    Default maximum ReAct turns (default 10).
DFTOOLBENCH_TEMPERATURE
    Default sampling temperature (default 0.0).
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK imports
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as _genai  # type: ignore
    _GENAI_AVAILABLE = True
except ImportError:
    _genai = None  # type: ignore
    _GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

_FORENSIC_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert digital forensics analyst specialising in image-based
    deepfake detection.  Your task is to answer the user's question about an
    image by systematically reasoning through the evidence and, where
    necessary, invoking the tools provided to you.

    ## Available tools
    {tool_schema}

    ## Instructions
    Follow the ReAct protocol strictly:

    Thought: <your reasoning about the current state>
    Action: <tool name, or "Final Answer">
    Action Input: <JSON object of arguments, or your final answer>

    Rules:
    - Always start with a Thought.
    - Use one Action per turn.
    - To finish, set Action to "Final Answer" and put your answer in Action Input.
    - Be concise, factual, and grounded in tool outputs.
    - When a tool returns an error, note it in your Thought and try an
      alternative approach or conclude with the available evidence.
    - Never fabricate tool outputs.
    """
)

_E2E_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert digital forensics analyst specialising in image-based
    deepfake detection.  Answer the user's question directly and concisely,
    drawing only on the information provided.  Do not invoke any tools.
    """
)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class AgentMode(str, Enum):
    """Operating mode for the ReAct controller."""

    AGENT = "agent"
    """Full ReAct loop with tool invocations."""

    E2E = "e2e"
    """End-to-end single-pass direct answer (no tools)."""


@dataclass
class TurnRecord:
    """A single turn in the ReAct conversation.

    Parameters
    ----------
    thought : str
        The model's reasoning text.
    action : str
        The tool name (or ``"Final Answer"``).
    action_input : Any
        Parsed tool arguments or the final answer string.
    observation : str
        The result returned by the tool, or empty for a final answer turn.
    raw_output : str
        Unprocessed model output for this turn.
    """

    thought: str
    action: str
    action_input: Any
    observation: str
    raw_output: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class AgentTrace:
    """Full trace of a completed ReAct episode.

    Parameters
    ----------
    question : str
        The original question posed to the agent.
    final_answer : str
        The agent's final answer.
    turns : list[TurnRecord]
        All intermediate ReAct turns.
    stopped_reason : str
        Why the episode ended: ``"final_answer"``, ``"max_turns"``,
        or ``"error"``.
    """

    question: str
    final_answer: str
    turns: list[TurnRecord] = field(default_factory=list)
    stopped_reason: str = "final_answer"

    @property
    def predicted_steps(self) -> list[dict[str, Any]]:
        """Return the list of tool calls in the format expected by metrics.py."""
        return [
            {"tool": t.action, "args": t.action_input}
            for t in self.turns
            if t.action not in ("Final Answer", "")
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "turns": [t.to_dict() for t in self.turns],
            "stopped_reason": self.stopped_reason,
        }


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

def build_tool_schema_text(tools: list[dict[str, Any]]) -> str:
    """Render a list of tool descriptors as a human-readable schema block.

    Parameters
    ----------
    tools : list[dict]
        Each dict should have:

        - ``"name"`` (str): tool name.
        - ``"description"`` (str): what the tool does.
        - ``"parameters"`` (dict, optional): JSON-schema style parameter dict.

    Returns
    -------
    str
        Formatted multi-line string injected into the system prompt.
    """
    if not tools:
        return "(No tools available)"
    lines: list[str] = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        if params:
            lines.append("Parameters (JSON):")
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any") if isinstance(pinfo, dict) else str(pinfo)
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
                lines.append(f"  - {pname} ({ptype}): {pdesc}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_react_output(text: str) -> tuple[str, str, Any]:
    """Parse a single ReAct model response.

    Extracts the *Thought*, *Action*, and *Action Input* fields from the
    model's free-form text output.

    Parameters
    ----------
    text : str
        Raw text output from the model.

    Returns
    -------
    tuple[str, str, Any]
        ``(thought, action, action_input)`` where *action_input* is a
        parsed JSON object when the text is valid JSON, otherwise a raw
        string.
    """
    thought = ""
    action = ""
    action_input_raw = ""

    # Extract Thought
    thought_match = re.search(
        r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract Action
    action_match = re.search(
        r"Action\s*:\s*(.+?)(?=\nAction\s*Input\s*:|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    if action_match:
        action = action_match.group(1).strip().strip('"').strip("'")

    # Extract Action Input
    input_match = re.search(
        r"Action\s*Input\s*:\s*(.+?)(?=\nThought\s*:|\nAction\s*:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if input_match:
        action_input_raw = input_match.group(1).strip()

    # Attempt JSON parse of action input
    action_input: Any = action_input_raw
    # Strip surrounding code fences if present
    fence = re.match(r"^```(?:json)?\s*([\s\S]+?)```\s*$", action_input_raw)
    if fence:
        action_input_raw = fence.group(1).strip()
    try:
        action_input = json.loads(action_input_raw)
    except (json.JSONDecodeError, TypeError):
        action_input = action_input_raw  # keep as string

    return thought, action, action_input


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

class _ClaudeBackend:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required for the Claude backend. "
                "Install it with: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
        self._client = _anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(
        self, system_prompt: str, messages: list[dict[str, str]]
    ) -> str:
        """Send a multi-turn conversation and return the assistant reply."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text


class _GeminiBackend:
    """Thin wrapper around the Google Generative AI SDK."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "The 'google-generativeai' package is required for the Gemini backend. "
                "Install it with: pip install google-generativeai"
            )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        _genai.configure(api_key=api_key)
        self._gen_config = _genai.GenerationConfig(  # type: ignore[attr-defined]
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._model = _genai.GenerativeModel(model_name=model)  # type: ignore[attr-defined]
        self._system_prompt: str = ""

    def complete(
        self, system_prompt: str, messages: list[dict[str, str]]
    ) -> str:
        """Send a multi-turn conversation and return the assistant reply."""
        # Gemini does not have a separate system turn in the same way;
        # prepend it as the first user message when it changes.
        history = []
        if system_prompt:
            # Gemini SDK uses "user"/"model" role names.
            history.append({"role": "user", "parts": [system_prompt]})
            history.append({"role": "model", "parts": ["Understood."]})

        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = self._model.start_chat(history=history[:-1] if history else [])
        last_content = history[-1]["parts"][0] if history else ""
        response = chat.send_message(
            last_content, generation_config=self._gen_config
        )
        return response.text


def _build_backend(
    model: str, temperature: float, max_tokens: int
) -> "_ClaudeBackend | _GeminiBackend":
    """Instantiate the correct backend based on the model name prefix."""
    if model.startswith("gemini"):
        return _GeminiBackend(model, temperature, max_tokens)
    return _ClaudeBackend(model, temperature, max_tokens)


# ---------------------------------------------------------------------------
# ReAct Controller
# ---------------------------------------------------------------------------

class ReActController:
    """Tool-augmented ReAct agent controller for deepfake forensics.

    Parameters
    ----------
    tools : list[dict], optional
        Tool descriptors (schema) available to the agent.  Each dict should
        have ``"name"``, ``"description"``, and optionally ``"parameters"``.
    tool_executor : callable, optional
        A callable with signature ``(tool_name: str, args: Any) -> str`` that
        executes a tool and returns its observation string.
    model : str, optional
        Model identifier.  Defaults to ``DFTOOLBENCH_MODEL`` env var or
        ``claude-3-5-sonnet-20241022``.
    temperature : float, optional
        Sampling temperature.  Defaults to ``DFTOOLBENCH_TEMPERATURE`` env var
        or 0.0.
    max_turns : int, optional
        Maximum ReAct turns before forcing a stop.  Defaults to
        ``DFTOOLBENCH_MAX_TURNS`` env var or 10.
    max_tokens : int, optional
        Maximum tokens per model response (default 2048).
    mode : AgentMode, optional
        Operating mode (default ``AgentMode.AGENT``).

    Examples
    --------
    >>> def my_executor(tool_name, args):
    ...     return f"Result of {tool_name}({args})"
    >>> ctrl = ReActController(
    ...     tools=[{"name": "OCR", "description": "Extract text from image."}],
    ...     tool_executor=my_executor,
    ... )
    >>> trace = ctrl.run("Does this image contain text?")
    >>> print(trace.final_answer)
    """

    def __init__(
        self,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, Any], str]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_turns: Optional[int] = None,
        max_tokens: int = 2048,
        mode: AgentMode = AgentMode.AGENT,
    ) -> None:
        self.tools: list[dict[str, Any]] = tools or []
        self.tool_executor: Callable[[str, Any], str] = tool_executor or self._noop_executor
        self.model: str = (
            model
            or os.environ.get("DFTOOLBENCH_MODEL", _DEFAULT_MODEL)
        )
        self.temperature: float = float(
            temperature
            if temperature is not None
            else os.environ.get("DFTOOLBENCH_TEMPERATURE", "0.0")
        )
        self.max_turns: int = int(
            max_turns
            if max_turns is not None
            else os.environ.get("DFTOOLBENCH_MAX_TURNS", "10")
        )
        self.max_tokens = max_tokens
        self.mode = AgentMode(mode)

        self._backend = _build_backend(self.model, self.temperature, self.max_tokens)
        self._tool_schema_text = build_tool_schema_text(self.tools)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _noop_executor(tool_name: str, args: Any) -> str:
        """Default executor — returns a placeholder when no executor is set."""
        return f"[Tool '{tool_name}' is not registered.  No output available.]"

    def _system_prompt(self) -> str:
        """Build the system prompt for the configured mode."""
        if self.mode == AgentMode.E2E:
            return _E2E_SYSTEM_PROMPT
        return _FORENSIC_SYSTEM_PROMPT.format(tool_schema=self._tool_schema_text)

    @staticmethod
    def _observation_message(turn: TurnRecord) -> str:
        """Format a turn's observation as a new user message."""
        return f"Observation: {turn.observation}"

    # ------------------------------------------------------------------
    # E2E mode
    # ------------------------------------------------------------------

    def _run_e2e(self, question: str) -> AgentTrace:
        """Run in end-to-end (single-pass) mode.

        Parameters
        ----------
        question : str
            The question to answer.

        Returns
        -------
        AgentTrace
            Trace with a single implicit turn and the model's direct answer.
        """
        messages = [{"role": "user", "content": question}]
        raw = self._backend.complete(self._system_prompt(), messages)
        return AgentTrace(
            question=question,
            final_answer=raw.strip(),
            turns=[
                TurnRecord(
                    thought="",
                    action="Final Answer",
                    action_input=raw.strip(),
                    observation="",
                    raw_output=raw,
                )
            ],
            stopped_reason="final_answer",
        )

    # ------------------------------------------------------------------
    # Agent mode (full ReAct loop)
    # ------------------------------------------------------------------

    def _run_agent(self, question: str) -> AgentTrace:
        """Run the full ReAct loop.

        Parameters
        ----------
        question : str
            The question to answer.

        Returns
        -------
        AgentTrace
            Complete episode trace.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": question}]
        turns: list[TurnRecord] = []
        final_answer = ""
        stopped_reason = "max_turns"
        system_prompt = self._system_prompt()

        for turn_idx in range(self.max_turns):
            logger.debug("ReAct turn %d/%d", turn_idx + 1, self.max_turns)

            try:
                raw = self._backend.complete(system_prompt, messages)
            except Exception as exc:  # noqa: BLE001
                logger.error("Model call failed on turn %d: %s", turn_idx + 1, exc)
                stopped_reason = "error"
                break

            thought, action, action_input = parse_react_output(raw)

            # Append assistant turn
            messages.append({"role": "assistant", "content": raw})

            if action.strip().lower() in ("final answer", "final_answer"):
                final_answer = (
                    action_input
                    if isinstance(action_input, str)
                    else json.dumps(action_input)
                )
                turns.append(
                    TurnRecord(
                        thought=thought,
                        action="Final Answer",
                        action_input=action_input,
                        observation="",
                        raw_output=raw,
                    )
                )
                stopped_reason = "final_answer"
                break

            # Execute tool
            observation = ""
            if action:
                try:
                    observation = self.tool_executor(action, action_input)
                except Exception as exc:  # noqa: BLE001
                    observation = f"[Error executing tool '{action}': {exc}]"
                    logger.warning("Tool '%s' raised: %s", action, exc)

            turn_record = TurnRecord(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                raw_output=raw,
            )
            turns.append(turn_record)

            # Feed observation back as a user message
            obs_msg = self._observation_message(turn_record)
            messages.append({"role": "user", "content": obs_msg})

        if not final_answer and turns:
            # Use the last action input as a best-effort answer
            last = turns[-1]
            final_answer = (
                last.action_input
                if isinstance(last.action_input, str)
                else json.dumps(last.action_input)
            )

        return AgentTrace(
            question=question,
            final_answer=final_answer,
            turns=turns,
            stopped_reason=stopped_reason,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> AgentTrace:
        """Run the agent on a single question.

        Parameters
        ----------
        question : str
            The forensic question to answer.

        Returns
        -------
        AgentTrace
            Full episode trace including the final answer.
        """
        if self.mode == AgentMode.E2E:
            return self._run_e2e(question)
        return self._run_agent(question)

    def run_batch(
        self, questions: list[str]
    ) -> list[AgentTrace]:
        """Run the agent on a batch of questions sequentially.

        Parameters
        ----------
        questions : list[str]
            Questions to answer.

        Returns
        -------
        list[AgentTrace]
            One trace per question, in the same order.
        """
        traces = []
        for i, q in enumerate(questions):
            logger.info("Processing question %d/%d", i + 1, len(questions))
            traces.append(self.run(q))
        return traces

    def update_tools(self, tools: list[dict[str, Any]]) -> None:
        """Replace the tool registry and rebuild the schema text.

        Parameters
        ----------
        tools : list[dict]
            New tool descriptors.
        """
        self.tools = tools
        self._tool_schema_text = build_tool_schema_text(tools)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:  # pragma: no cover
    """CLI entry point for single-question interactive testing.

    Usage::

        python -m dftoolbench.evaluation.react_controller \\
            --question "Is this image a deepfake?" \\
            [--tools tools.json] [--model <name>] [--mode agent|e2e]
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="DFToolBench-I ReAct agent controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="Question to ask the agent.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help="Path to a JSON file containing a list of tool descriptors.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (overrides DFTOOLBENCH_MODEL).",
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "e2e"],
        default="agent",
        help="Operating mode.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum ReAct turns.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path.  Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tools: list[dict] = []
    if args.tools:
        with open(args.tools, encoding="utf-8") as fh:
            tools = json.load(fh)

    ctrl = ReActController(
        tools=tools,
        model=args.model,
        max_turns=args.max_turns,
        mode=AgentMode(args.mode),
    )

    trace = ctrl.run(args.question)
    json_str = json.dumps(trace.to_dict(), indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"Trace written to {args.out}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    _main()
