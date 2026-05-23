import asyncio
import streamlit as st
from stockbot.core import AgentCore
from stockbot.quota import QuotaManager
from stockbot.memory.profile import ProfileManager
from stockbot.events import TextDelta, ToolCallStart, ToolCallEnd, Error, QuotaExceeded


def render_chat(agent: AgentCore, quota: QuotaManager, profile: ProfileManager, user: dict):
    user_id = user["id"]

    # ── Sidebar ──
    with st.sidebar:
        st.title("StockBot")

        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Watchlist")
        favs = profile.get_favorites(user_id)
        if favs:
            for sym in favs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"{sym}", key=f"fav_{sym}", use_container_width=True):
                        st.session_state.pending_input = f"Analyze {sym}"
                        st.rerun()
                with col2:
                    if st.button("x", key=f"rm_{sym}"):
                        profile.remove_favorite(user_id, sym)
                        st.rerun()
        else:
            st.caption("No watched stocks yet")

        st.divider()
        qr = quota.check(user_id)
        st.metric("Daily Usage", f"{qr.used}/{qr.limit}", delta=f"{qr.remain} remaining")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Market", use_container_width=True):
                st.session_state.pending_input = "How is the market today?"
                st.rerun()
        with c2:
            if st.button("Hot Sectors", use_container_width=True):
                st.session_state.pending_input = "What sectors are hot recently?"
                st.rerun()

        st.divider()
        if st.button("Logout", use_container_width=True):
            del st.session_state["user"]
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()

    # ── Main chat ──
    st.caption(f"Logged in as: {user['username']}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_input" in st.session_state and st.session_state.pending_input:
        prompt = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        prompt = None

    # Welcome for new users
    if not st.session_state.messages:
        st.markdown("""
        ### Welcome to StockBot!

        I can help you with:
        - Real-time stock quotes
        - Technical analysis (trend with scenarios)
        - Financial data interpretation
        - Stock news search

        Try asking me:

        """)
        examples = [
            "What sectors are hot recently?",
            "Analyze Kweichow Moutai 600519",
            "Find high-dividend bank stocks",
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            with cols[i]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    prompt = ex

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    input_text = st.chat_input("Ask about stocks...")

    if prompt:
        input_text = prompt

    if input_text:
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.markdown(input_text)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            response_text = ""

            async def stream():
                nonlocal response_text
                async for evt in agent.run(user_id, input_text):
                    if isinstance(evt, QuotaExceeded):
                        st.error(f"Daily quota exhausted ({evt.used}/{evt.limit}). Resets tomorrow.")
                        return
                    if isinstance(evt, TextDelta):
                        response_text += evt.content
                        response_placeholder.markdown(response_text)
                    if isinstance(evt, ToolCallStart):
                        status_placeholder.info(f"Running {evt.name}...")
                    if isinstance(evt, ToolCallEnd):
                        status_placeholder.success(f"{evt.name} completed")
                    if isinstance(evt, Error):
                        st.error(f"Error: {evt.message}")
                        return
                status_placeholder.empty()

            asyncio.run(stream())

            if response_text:
                disclaimer = "\n\n---\n*Analysis for reference only. Not investment advice.*"
                response_placeholder.markdown(response_text + disclaimer)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
