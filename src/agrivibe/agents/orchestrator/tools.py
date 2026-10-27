"""The specialists, wrapped as tools. The orchestrator calls an agent the way it calls any tool.

This is LangChain's subagents pattern: each tool runs its specialist with `ainvoke` and
hands back the answer. The tools also connect the specialists to central memory (see
memory.py): they read the saved requirements into the specialist's task and write the
specialist's report to the findings.
"""

import json
import logging

from langchain.messages import ToolMessage
from langchain.tools import BaseTool, ToolRuntime, tool
from langgraph.types import Command

from agrivibe.agents.fieldsat import FieldContext
from agrivibe.agents.orchestrator.memory import AgriVibeState, FarmerContext, Finding, Requirements

logger = logging.getLogger(__name__)


def brief(task: str, requirements: Requirements | None) -> str:
    """Puts the saved requirements above the task, so every specialist works on the same field."""
    if not requirements:
        return task
    return (
        f"Field: {requirements['field_id']} ({requirements['crop']})\n"
        f"Farmer's problem: {requirements['problem']}\n"
        f"Time frame: {requirements['timeframe']}\n\n"
        f"{task}"
    )


def as_text(finding: Finding) -> str:
    """Writes a finding for a model to read: the summary, then the data as JSON."""
    if not finding["data"]:
        return finding["summary"]
    return f"{finding['summary']}\nData: {json.dumps(finding['data'])}"


def failed(error: Exception) -> Finding:
    """The finding for a specialist that raised, so one broken specialist does not end the whole answer."""
    return Finding(summary=f"This agent failed and returned nothing ({error}).", data={})


def save_finding(name: str, finding: Finding, runtime: ToolRuntime) -> Command:
    """Writes a specialist's finding to central memory and hands it to the orchestrator.

    A tool that updates state returns a Command instead of a string. The Command must
    include the ToolMessage the orchestrator reads as the tool's result.
    """
    return Command(update={
        "findings": {name: finding},
        "messages": [ToolMessage(as_text(finding), tool_call_id=runtime.tool_call_id)],
    })


@tool(parse_docstring=True)
def save_requirements(
    field_id: str, crop: str, problem: str, timeframe: str, runtime: ToolRuntime[FarmerContext, AgriVibeState]
) -> Command:
    """Save what the farmer needs. Call it once you know the field, before you ask a specialist about it.

    Args:
        field_id: The field's ID from list_fields, for example SL-PADDY-MINNERIYA.
        crop: The crop on the field, for example paddy.
        problem: The farmer's question or problem, in a few words.
        timeframe: The period it is about, for example "this week". Use "now" if the farmer gave none.
    """
    requirements = Requirements(field_id=field_id, crop=crop, problem=problem, timeframe=timeframe)
    # Long-term memory: the farmer's field is still known in their next conversation.
    if runtime.store is not None:
        runtime.store.put(("farmers", runtime.context.user_id), "farm", {"field_id": field_id, "crop": crop})
    return Command(update={
        "requirements": requirements,
        "messages": [ToolMessage(f"Saved: {requirements}", tool_call_id=runtime.tool_call_id)],
    })


def make_specialist_tools(weather_agent, farm_context_agent, knowledge_agent, advisory_agent) -> list[BaseTool]:
    """Wraps each specialist agent as a tool for the orchestrator.

    The docstrings below are what the orchestrator reads when it decides whom to call.
    Each specialist starts with an empty conversation: it sees the task, not the chat
    with the farmer.

    Args:
        weather_agent: Built by build_weather_agent.
        farm_context_agent: Built by build_farm_context_agent.
        knowledge_agent: Built by build_fieldoracle_agent, without a checkpointer.
        advisory_agent: Built by build_advisory_agent.

    Returns:
        ask_weather_agent, ask_farm_context_agent, ask_knowledge_agent and ask_advisory_agent.
    """

    async def ask_fieldsat_agent(agent, name: str, task: str, runtime: ToolRuntime[FarmerContext, AgriVibeState]):
        """Runs the Weather or Farm Context Agent on the saved field and saves its report as a finding."""
        requirements = runtime.state.get("requirements")
        if not requirements:
            finding = Finding(summary="No field is saved yet. Call save_requirements first.", data={})
            return save_finding(name, finding, runtime)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": brief(task, requirements)}]},
                context=FieldContext(field_id=requirements["field_id"]),
            )
            report = result["structured_response"]  # a FieldReport
            finding = Finding(summary=report.summary, data=report.data or {})
        except Exception as error:
            logger.exception("%s failed", agent.name)
            finding = failed(error)
        return save_finding(name, finding, runtime)

    @tool(parse_docstring=True)
    async def ask_weather_agent(task: str, runtime: ToolRuntime[FarmerContext, AgriVibeState]) -> Command:
        """Ask the Weather Agent about the rain on the farmer's field: the last 7 days, the last 48 hours and the next 48 hours.

        Use it for the state of a field and for timing field work such as spraying or fertilising.

        Args:
            task: What you need, for example "Rain in the last week and the next 48 hours". The field
                comes from the saved requirements.
        """
        return await ask_fieldsat_agent(weather_agent, "weather", task, runtime)

    @tool(parse_docstring=True)
    async def ask_farm_context_agent(task: str, runtime: ToolRuntime[FarmerContext, AgriVibeState]) -> Command:
        """Ask the Farm Context Agent how healthy the farmer's field looks from the satellite: NDVI trend, uniformity and data quality.

        Use it for questions about the state of a field, such as stress or poor growth.

        Args:
            task: What you need, for example "NDVI trend over the last 14 days". The field comes from
                the saved requirements.
        """
        return await ask_fieldsat_agent(farm_context_agent, "farm_context", task, runtime)

    @tool(parse_docstring=True)
    async def ask_knowledge_agent(question: str, runtime: ToolRuntime[FarmerContext, AgriVibeState]) -> Command:
        """Ask the Agri Knowledge Agent (FieldOracle), which searches cultivation guides, government policies and farm records.

        Use it for rates, varieties, schedules, subsidies and what is recorded about a farm. Its answer
        cites passage ids in brackets.

        Args:
            question: A complete question that makes sense on its own, for example "When is the first
                top dressing due for Bg 300 sown on 2 October 2026?"
        """
        task = brief(question, runtime.state.get("requirements"))
        try:
            result = await knowledge_agent.ainvoke({"messages": [{"role": "user", "content": task}]})
            answer = result["structured_response"]  # a CitedAnswer
            # Each sentence followed by the passages it cites, and every cited id once, in order.
            text = " ".join(f"{s.sentence} [{', '.join(s.citations)}]" for s in answer.sentences)
            citations = list(dict.fromkeys(c for s in answer.sentences for c in s.citations))
            finding = Finding(
                summary=text if answer.found and text else "The knowledge base does not cover this.",
                data={"found": answer.found, "citations": citations},
            )
        except Exception as error:
            logger.exception("%s failed", knowledge_agent.name)
            finding = failed(error)
        return save_finding("knowledge", finding, runtime)

    # return_direct: the Advisory Agent's answer is the reply. The orchestrator does not reword it.
    @tool(parse_docstring=True, return_direct=True)
    async def ask_advisory_agent(decision: str, runtime: ToolRuntime[FarmerContext, AgriVibeState]) -> str:
        """Ask the Advisory Agent for the final recommendation on a decision, such as whether to spray or fertilise this week.

        Call it last and on its own, after the specialists the decision needs. It reads their findings
        from central memory, and its answer goes straight to the farmer.

        Args:
            decision: The decision in one sentence, for example "Should the farmer spray fungicide this week?"
        """
        # The Advisory Agent has no tools: every finding in central memory goes into its message.
        findings = runtime.state.get("findings") or {}
        evidence = "\n\n".join(f"{name}:\n{as_text(finding)}" for name, finding in findings.items())
        task = brief(
            f"Decision to make: {decision}\n\nEvidence:\n{evidence or 'None. No specialist was asked.'}",
            runtime.state.get("requirements"),
        )
        try:
            result = await advisory_agent.ainvoke({"messages": [{"role": "user", "content": task}]})
            return result["messages"][-1].text
        except Exception as error:
            logger.exception("%s failed", advisory_agent.name)
            return failed(error)["summary"]

    return [ask_weather_agent, ask_farm_context_agent, ask_knowledge_agent, ask_advisory_agent]
