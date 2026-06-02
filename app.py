#!/usr/bin/env python
"""Web entry point for StockBot — Streamlit multi-user app."""
import streamlit as st
from stockbot import create_agent
from stockbot.admin import AdminService
from stockbot.ui.login_page import render_login
from stockbot.ui.chat_page import render_chat
from stockbot.ui.admin_page import render_admin


def main():
    st.set_page_config(page_title="StockBot — A股智能投资分析助手", page_icon="📈", layout="wide")

    if "agent" not in st.session_state:
        agent, store, auth, cfg = create_agent()
        st.session_state.agent = agent
        st.session_state.store = store
        st.session_state.auth = auth
        st.session_state.cfg = cfg
        admin_svc = AdminService(store, agent.quota, auth)
        st.session_state.admin_svc = admin_svc

    auth = st.session_state.auth

    if "user" not in st.session_state:
        render_login(auth)
    else:
        user = st.session_state.user

        if user.get("role") == "admin":
            page = st.sidebar.radio("Navigation", ["Chat", "Admin"])
        else:
            page = "Chat"

        if page == "Chat":
            render_chat(st.session_state.agent, st.session_state.agent.quota,
                        st.session_state.agent.profile, user)
        elif page == "Admin":
            render_admin(st.session_state.admin_svc, st.session_state.agent.quota)


if __name__ == "__main__":
    main()
