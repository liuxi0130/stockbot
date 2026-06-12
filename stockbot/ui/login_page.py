import streamlit as st
import streamlit.components.v1 as components
from stockbot.auth import AuthManager


COOKIE_NAME = "stockbot_session"


def _set_session_cookie(token: str):
    """Inject JS to set a persistent cookie in the browser (30 days)."""
    components.html(f"""
    <script>
    (function() {{
        var d = new Date();
        d.setTime(d.getTime() + (30 * 24 * 60 * 60 * 1000));
        document.cookie = "{COOKIE_NAME}=" + "{token}" + ";expires=" + d.toUTCString() + ";path=/;SameSite=Lax";
    }})();
    </script>
    """, height=0)


def _clear_session_cookie():
    """Inject JS to clear the session cookie."""
    components.html(f"""
    <script>
    (function() {{
        document.cookie = "{COOKIE_NAME}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;SameSite=Lax";
    }})();
    </script>
    """, height=0)


def render_login(auth: AuthManager):
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1>📈 StockBot</h1>
        <p style="color: #666; font-size: 1.1rem;">A 股智能投资分析助手</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["登录", "注册"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pass")
            submitted = st.form_submit_button("登   录", use_container_width=True)
            if submitted:
                user = auth.login(username, password)
                if user:
                    st.session_state["user"] = user
                    # Create persistent session token and set cookie
                    token = auth.create_session(user["id"])
                    st.session_state["session_token"] = token
                    _set_session_cookie(token)
                    store = st.session_state.get("store")
                    if store:
                        store.log_activity(user["id"], "login", "Web 登录成功")
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("用户名", key="reg_user")
            new_pass = st.text_input("密码", type="password", key="reg_pass")
            confirm = st.text_input("确认密码", type="password", key="reg_confirm")
            submitted = st.form_submit_button("注   册", use_container_width=True)
            if submitted:
                if not new_user or not new_pass:
                    st.error("请填写所有字段")
                elif new_pass != confirm:
                    st.error("两次密码不一致")
                else:
                    uid = auth.register(new_user, new_pass)
                    if uid is None:
                        st.error("用户名已存在或注册已关闭")
                    else:
                        st.success("注册成功！请切换到登录页签进行登录。")
