from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.types import Command
from langgraph.config import get_stream_writer
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from typing import Annotated, Any
from llm import get_llm

llm = get_llm()

# ---------------------------------------------------------------
# Tool 1: Analyze Meeting
# ---------------------------------------------------------------
@tool
def analyze_meeting(
    transcript: str,
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
) -> str:
    """Analyze the meeting transcript."""

    if state.get("processed"):
        return "Transcript already analyzed in this session."

    ctx = config.get("configurable", {})
    team = ctx.get("team", "General")
    project = ctx.get("project", "Unknown")

    if len(transcript.strip()) < 50:
        raise ValueError("Transcript too short.")

    writer = get_stream_writer()
    writer(f"🔍 Analyzing transcript for {team} / {project}...\n")

    prompt = f"""
Analyze this meeting transcript for team {team}, project {project}:

{transcript}

Return JSON with:
- summary
- action_items
- decisions
- discussion_points
"""

    response = llm.invoke(prompt)
    writer("✅ Analysis complete.\n")
    return response.content


# ---------------------------------------------------------------
# Tool 2: Assign Owners
# ---------------------------------------------------------------
@tool
def assign_owners(
    action_items: str,
    config: RunnableConfig,
) -> str:
    """Assign owners to action items."""

    ctx = config.get("configurable", {})
    team = ctx.get("team", "General")

    if not action_items.strip():
        raise ValueError("No action items.")

    prompt = f"""
Assign owners to these action items from team {team}:

{action_items}

Return JSON list of:
{{ "task": "...", "owner": "...", "priority": "high|medium|low" }}
"""

    return llm.invoke(prompt).content


# ---------------------------------------------------------------
# Tool 3: Persist Results  ✅ WORKING VERSION
# ---------------------------------------------------------------
@tool
def persist_results(
    final_summary: str,
    store: Annotated[Any, InjectedStore],
    config: RunnableConfig,
) -> Command:
    """Persist results into long-term store."""

    ctx = config.get("configurable", {})
    project = ctx["project"]
    thread_id = ctx["thread_id"]

    namespace = ("meetings", project)

    store.put(
        namespace,
        thread_id,
        {
            "analysis": final_summary,
            "project": project,
            "thread": thread_id
        },
    )

    tool_call_id = config["invocation_params"]["tool_call_id"]

    return Command(
        messages=[
            ToolMessage(
                content="✅ Results persisted successfully.",
                tool_call_id=tool_call_id
            )
        ],
        update={
            "processed": True,
            "stored": True
        }
    )


tools = [analyze_meeting, assign_owners, persist_results]