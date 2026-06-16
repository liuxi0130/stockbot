#!/usr/bin/env python
"""Web entry point for StockBot — Streamlit multi-user app."""
import streamlit as st
import streamlit.components.v1 as components
from stockbot import create_agent
from stockbot.admin import AdminService
from stockbot.ui.login_page import render_login, COOKIE_NAME, _set_session_cookie, _clear_session_cookie
from stockbot.ui.chat_page import render_chat
from stockbot.ui.admin_page import render_admin
from stockbot.ui.worldcup_page import render_worldcup


def _check_persistent_session(auth):
    """Try to restore login from a persistent cookie or query param."""
    token = None

    # 1. Try reading the session cookie (Streamlit >= 1.37)
    try:
        cookies = st.context.cookies
        token = cookies.get(COOKIE_NAME)
    except AttributeError:
        pass

    # 2. Fallback: try reading cookie via JS interop (stored in session_state from previous loads)
    if not token:
        token = st.session_state.get("pending_cookie_token")

    # 3. Validate the token
    if token:
        user = auth.validate_session(token)
        if user:
            st.session_state["user"] = user
            st.session_state["session_token"] = token
            # Refresh cookie expiry on each visit
            _set_session_cookie(token)
            return True
        else:
            # Token is invalid/expired — clean up
            if "pending_cookie_token" in st.session_state:
                del st.session_state["pending_cookie_token"]
            _clear_session_cookie()

    return False


def _inject_cookie_reader():
    """Inject JS that reads the cookie and passes it back via query param.
    This runs once on first load when st.context.cookies is unavailable."""
    components.html(f"""
    <script>
    (function() {{
        var cookies = document.cookie.split('; ');
        for (var i = 0; i < cookies.length; i++) {{
            var parts = cookies[i].split('=');
            if (parts[0] === '{COOKIE_NAME}' && parts[1]) {{
                var url = new URL(window.location);
                if (!url.searchParams.has('_s')) {{
                    url.searchParams.set('_s', parts[1]);
                    window.location.href = url.toString();
                    break;
                }}
            }}
        }}
    }})();
    </script>
    """, height=0)


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

    # Check query param fallback for cookie reading (pre-st.context.cookies Streamlit versions)
    query_params = st.query_params
    if "_s" in query_params and "user" not in st.session_state:
        st.session_state["pending_cookie_token"] = query_params["_s"]

    if "user" not in st.session_state:
        # Attempt persistent session restore
        if not _check_persistent_session(auth):
            # Show login page; inject cookie reader for fallback support
            try:
                st.context.cookies
            except AttributeError:
                if "_s" not in query_params:
                    _inject_cookie_reader()
            render_login(auth)
            st.stop()

    # User is logged in
    if "_s" in query_params:
        st.query_params.clear()

    user = st.session_state.user

    if user.get("role") == "admin":
        page = st.sidebar.radio("Navigation", ["💬 Chat", "⚽ 世界杯", "🔧 Admin"])
    else:
        page = st.sidebar.radio("Navigation", ["💬 Chat", "⚽ 世界杯"])

    if page == "💬 Chat":
        render_chat(st.session_state.agent, st.session_state.agent.quota,
                    st.session_state.agent.profile, user)
    elif page == "⚽ 世界杯":
        render_worldcup()
    elif page == "🔧 Admin":
        render_admin(st.session_state.admin_svc, st.session_state.agent.quota)


if __name__ == "__main__":
    main()
