from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from llm import get_llm
from tools import tools
from state import MeetingAgentState
from memory import store, checkpointer

llm = get_llm()

# ── Agent (ReAct loop lives inside create_react_agent) ────────────
# create_react_agent is the modern equivalent of AgentExecutor.
# It handles: tool calling, observation feeding,
# and the full Reason → Act → Observe cycle automatically.
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_schema=MeetingAgentState,
    store=store,
    checkpointer=checkpointer,
)


def _stream_agent(user_message: HumanMessage, config: dict) -> dict:
    """
    Internal: streams agent execution and separates
    custom stream_writer events from state messages.

    stream_mode=["values", "custom"] gives us:
      - "values" chunks -> full state snapshots (ReAct loop steps)
      - "custom" chunks -> stream_writer output from inside tools
    """
    collected     = []
    stream_events = []    # captures runtime.stream_writer output

    for stream_mode, chunk in agent.stream(
        {"messages": [user_message]},
        config=config,
        stream_mode=["values", "custom"],
    ):
        if stream_mode == "custom":
            # runtime.stream_writer - live progress pushed from inside tools
            stream_events.append(chunk)

        elif stream_mode == "values":
            messages = chunk.get("messages", [])
            if messages:
                last    = messages[-1]
                content = getattr(last, "content", "")
                label   = last.__class__.__name__
                if content:
                    collected.append({
                        "role":    label,
                        "content": content,
                    })

    return {"messages": collected, "stream_events": stream_events}


def run_agent(transcript: str, config: dict) -> dict:
    """
    Initial run - analyzes transcript, assigns owners, persists results.
    Uses a fresh thread_id so state starts clean for every new analysis.
    """

    user_message = HumanMessage(content=f"""
Analyze this meeting transcript by following these steps in order:
1. Call analyze_meeting with the transcript
2. Call assign_owners with the action items from step 1
3. Call persist_results with the complete final analysis

Transcript:
{transcript}
""")

    result = _stream_agent(user_message, config)

    # Read back from runtime.store to confirm persistence
    project     = config.get("configurable", {}).get("project", "general")
    saved       = store.search(("meetings", project))
    stored_data = [{"key": item.key, "value": item.value} for item in saved]

    return {
        "messages":      result["messages"],
        "stream_events": result["stream_events"],
        "stored_data":   stored_data,
    }


def run_followup(question: str, config: dict) -> dict:
    """
    Follow-up run - uses the SAME thread_id as the initial analysis.

    Because MemorySaver checkpoints full state under thread_id:
      - Agent loads previous state -> processed = True is already set
      - If LLM tries to call analyze_meeting -> guard triggers -> skips it
      - Agent answers directly from existing conversation history

    This is where runtime.state actually earns its place.
    """

    user_message = HumanMessage(content=f"""
The meeting has already been analyzed in this session.
Do NOT call analyze_meeting again.
Answer this follow-up question using the analysis already done:

{question}
""")

    result = _stream_agent(user_message, config)

    return {
        "messages":      result["messages"],
        "stream_events": result["stream_events"],
    }
