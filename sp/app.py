import streamlit as st
import json
import uuid
from agent import run_agent, run_followup

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Meeting Intelligence Agent",
    layout="wide",
)

# ── Session State Init ────────────────────────────────────────────
# thread_id is generated ONCE per analysis and reused for follow-ups.
# This is what makes runtime.state meaningful —
# the same thread carries processed=True into every follow-up call.
if "thread_id"       not in st.session_state:
    st.session_state.thread_id       = None
if "analysis_done"   not in st.session_state:
    st.session_state.analysis_done   = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "followups"       not in st.session_state:
    st.session_state.followups       = []   # list of {question, answer}
if "config"          not in st.session_state:
    st.session_state.config          = None


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Session Config")
    st.caption("Passed as `runtime.context` — immutable at invocation time")

    team     = st.text_input("Team",    value="Backend")
    project  = st.text_input("Project", value="GenAI Gateway")
    language = st.text_input("Language",value="en")

    st.divider()

    if st.session_state.thread_id:
        st.markdown("**Active Thread ID**")
        st.code(st.session_state.thread_id, language=None)
        st.caption("Same thread_id is reused for follow-ups so `processed=True` persists across calls.")

    st.divider()
    st.markdown("**Concepts in this project**")
    st.markdown("""
- ✅ `@tool` decorator
- ✅ `handle_tool_error=True`
- ✅ Tool schemas (auto-generated)
- ✅ `AgentExecutor` / ReAct loop
- ✅ `runtime.state` — guard in follow-ups
- ✅ `runtime.context` — sidebar config
- ✅ `runtime.store` — persistent memory
- ✅ `runtime.stream_writer` — progress log
- ✅ `Command` — state update signal
""")


# ── Title ─────────────────────────────────────────────────────────
st.title("🧠 Meeting Intelligence Agent")
st.caption("ReAct Agent · LangChain Tools · Persistent Memory · Multi-turn")

st.divider()

# ── Phase 1: Transcript Input ─────────────────────────────────────
if not st.session_state.analysis_done:

    st.subheader("📋 Paste Meeting Transcript")

    sample = """John: Let's start the GenAI Gateway planning meeting.

Sarah: The backend APIs are ready. We just need to finalize the auth layer.

John: Sarah, can you own the auth layer completion by Friday?

Sarah: Yes, I'll handle that.

Mike: I'll take care of the deployment scripts and CI/CD pipeline. Done by Thursday.

John: We decided to switch from in-memory storage to Redis for caching.

Sarah: Agreed. Also the API documentation needs updating before the release.

Mike: I can handle the docs as well.

John: Perfect. Let's review progress next Monday."""

    transcript = st.text_area(
        "Transcript",
        value=sample,
        height=280,
        label_visibility="collapsed",
    )

    if st.button("🚀 Analyze Meeting", type="primary", use_container_width=True):
        if len(transcript.strip()) < 50:
            st.error("Transcript too short. Please paste a complete meeting transcript.")
        else:
            # Generate a fresh thread_id for this analysis.
            # This ensures state starts clean — not carrying over
            # processed=True from a previous analysis run.
            new_thread_id = str(uuid.uuid4())

            config = {
                "configurable": {
                    "thread_id": new_thread_id,
                    "team":      team,
                    "project":   project,
                    "language":  language,
                }
            }

            st.session_state.thread_id = new_thread_id
            st.session_state.config    = config

            with st.spinner("🤖 Agent running... ReAct loop in progress"):
                result = run_agent(transcript, config)

            st.session_state.analysis_result = result
            st.session_state.analysis_done   = True
            st.rerun()


# ── Phase 2: Results + Follow-up ─────────────────────────────────
else:
    result = st.session_state.analysis_result
    messages    = result.get("messages", [])
    stored_data = result.get("stored_data", [])
    stream_events = result.get("stream_events", [])

    # ── Stream Writer Log ─────────────────────────────────────────
    if stream_events:
        with st.expander("📡 runtime.stream_writer — Live tool progress", expanded=False):
            for event in stream_events:
                st.text(event)

    # ── Parse results from messages ───────────────────────────────
    analysis_data = None
    owners_data   = None

    for msg in messages:
        if msg["role"] == "ToolMessage":
            try:
                parsed = json.loads(msg["content"])
                if isinstance(parsed, dict) and "action_items" in parsed:
                    analysis_data = parsed
                elif isinstance(parsed, list) and parsed and "owner" in str(parsed[0]):
                    owners_data = parsed
            except Exception:
                pass

    # ── Results Tabs ──────────────────────────────────────────────
    st.subheader("📊 Meeting Analysis")

    if analysis_data:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Summary",
            "✅ Action Items",
            "🗳️ Decisions",
            "💬 Discussion Points",
        ])

        with tab1:
            st.write(analysis_data.get("summary", "—"))

        with tab2:
            for i, item in enumerate(analysis_data.get("action_items", []), 1):
                st.markdown(f"**{i}.** {item}")

        with tab3:
            for d in analysis_data.get("decisions", []):
                st.markdown(f"- {d}")

        with tab4:
            for p in analysis_data.get("discussion_points", []):
                st.markdown(f"- {p}")

    # ── Owners Table ──────────────────────────────────────────────
    if owners_data:
        st.subheader("👥 Assigned Owners")
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}

        for item in owners_data:
            task     = item.get("task", "")
            owner    = item.get("owner", "Unassigned")
            priority = item.get("priority", "medium").lower()
            icon     = priority_icon.get(priority, "⚪")
            st.markdown(f"{icon} **{task}** → `{owner}` ({priority})")

    # ── Persistent Store ──────────────────────────────────────────
    if stored_data:
        with st.expander("💾 runtime.store — Persistent Memory", expanded=False):
            st.caption("Saved via `store.put()` inside `persist_results` tool")
            for item in stored_data:
                st.markdown(f"**Key:** `{item['key']}`")
                st.json(item["value"])

    st.divider()

    # ── Follow-up Chat ────────────────────────────────────────────
    # Uses the SAME thread_id → MemorySaver loads state with processed=True
    # If LLM tries to re-call analyze_meeting → guard in tool blocks it
    st.subheader("💬 Follow-up Questions")
    st.caption(
        "Uses the same `thread_id` — `processed=True` is already in state. "
        "The agent answers directly without re-analyzing the transcript."
    )

    # Show previous follow-up exchanges
    for qa in st.session_state.followups:
        with st.chat_message("user"):
            st.write(qa["question"])
        with st.chat_message("assistant"):
            st.write(qa["answer"])

    # Follow-up input
    followup_question = st.chat_input("Ask a follow-up about this meeting...")

    if followup_question:
        with st.chat_message("user"):
            st.write(followup_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                followup_result = run_followup(
                    followup_question,
                    st.session_state.config,
                )

            # Extract final AI answer
            answer = ""
            for msg in reversed(followup_result["messages"]):
                if msg["role"] == "AIMessage" and msg["content"]:
                    answer = msg["content"]
                    break

            st.write(answer if answer else "Could not generate a response.")

        st.session_state.followups.append({
            "question": followup_question,
            "answer":   answer,
        })

    st.divider()

    # ── New Analysis Button ───────────────────────────────────────
    if st.button("🔄 Start New Analysis", use_container_width=True):
        # Reset everything — next run gets a fresh thread_id
        st.session_state.thread_id       = None
        st.session_state.analysis_done   = False
        st.session_state.analysis_result = None
        st.session_state.followups       = []
        st.session_state.config          = None
        st.rerun()
