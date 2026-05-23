import streamlit as st
from stockbot.auth import AuthManager


def render_login(auth: AuthManager):
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1>StockBot</h1>
        <p style="color: #666; font-size: 1.1rem;">A-Share AI Analysis Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                user = auth.login(username, password)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("Username", key="reg_user")
            new_pass = st.text_input("Password", type="password", key="reg_pass")
            confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submitted = st.form_submit_button("Register", use_container_width=True)
            if submitted:
                if not new_user or not new_pass:
                    st.error("Please fill in all fields")
                elif new_pass != confirm:
                    st.error("Passwords do not match")
                else:
                    uid = auth.register(new_user, new_pass)
                    if uid is None:
                        st.error("Username already exists or registration is closed")
                    else:
                        st.success("Registration successful! Switch to Login tab to sign in.")
