"""A minimal tool-using agent.

This is the agent loop - the single most important concept in agentic AI:

    1. Send the conversation (plus the list of available tools) to the model.
    2. Look at what came back.
         - Plain text?      -> the model is done. Return the answer.
         - Function calls?  -> the model wants a tool run.
    3. Run the requested function in YOUR code.
    4. Append both the model's request and your result to the conversation.
    5. Go back to step 1.

The model never runs anything itself. It only ever *asks*. Every side effect
happens in your process, which is what makes agents auditable and safe.

Phase 4 replaces this hand-written loop with LangGraph, which adds state,
branching and retries on top of the same idea.
"""

from __future__ import annotations

from google.genai import types

from app.llm import LLMError, extract_text, generate
from app.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

SYSTEM_PROMPT = (
    "You are a precise research analyst. "
    "You have tools available - use them rather than guessing. "
    "Never perform arithmetic yourself; always call the calculate tool. "
    "When you have everything you need, give a clear final answer and show "
    "which numbers you computed."
)

# Safety rail: without a cap, a confused model can loop forever, and on a free
# tier that burns your daily quota in seconds.
MAX_STEPS = 6


def _run_tool(call: types.FunctionCall) -> dict:
    """Execute one tool call and return a JSON-serialisable result.

    A failing tool must NOT crash the agent. We hand the error back to the
    model as the tool's result so it can correct itself and try again -
    self-correction is a large part of what makes an agent useful.
    """
    function = TOOL_FUNCTIONS.get(call.name)
    if function is None:
        return {"error": f"No such tool: {call.name}"}

    arguments = dict(call.args or {})
    try:
        return {"result": function(**arguments)}
    except Exception as error:  # noqa: BLE001 - deliberately broad
        return {"error": f"{type(error).__name__}: {error}"}


def run(question: str, verbose: bool = True) -> str:
    """Answer a question, calling tools as needed.

    Args:
        question: What to ask the agent.
        verbose: Print each tool call so you can watch the loop work.

    Returns:
        The agent's final text answer.

    Raises:
        LLMError: If the API fails, or the loop hits MAX_STEPS.
    """
    # The full conversation. We resend all of it on every request, because
    # the API is stateless - it remembers nothing between calls.
    conversation: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    for step in range(1, MAX_STEPS + 1):
        response = generate(conversation, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS)

        candidate = response.candidates[0]
        parts = candidate.content.parts or []
        calls = [part.function_call for part in parts if part.function_call]

        # No tool requested -> the model has finished reasoning.
        if not calls:
            return extract_text(response)

        # Record the model's turn verbatim before answering it.
        conversation.append(candidate.content)

        results: list[types.Part] = []
        for call in calls:
            outcome = _run_tool(call)
            if verbose:
                arguments = dict(call.args or {})
                print(f"  [step {step}] {call.name}({arguments}) -> {outcome}")

            results.append(
                types.Part.from_function_response(name=call.name, response=outcome)
            )

        # Tool results go back as a 'user' turn - your program is speaking
        # on behalf of the tools.
        conversation.append(types.Content(role="user", parts=results))

    raise LLMError(
        f"Agent did not finish within {MAX_STEPS} steps. "
        "The model may be stuck in a tool-calling loop."
    )
