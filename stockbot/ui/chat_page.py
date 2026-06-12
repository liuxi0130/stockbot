import asyncio
import streamlit as st
import streamlit.components.v1 as components
from stockbot.core import AgentCore
from stockbot.quota import QuotaManager
from stockbot.memory.profile import ProfileManager
from stockbot.events import TextDelta, ToolCallStart, ToolCallEnd, Error, QuotaExceeded
from stockbot.ui.login_page import COOKIE_NAME, _clear_session_cookie


def render_chat(agent: AgentCore, quota: QuotaManager, profile: ProfileManager, user: dict):
    user_id = user["id"]

    # ── 侧边栏 ──
    with st.sidebar:
        st.title("📈 StockBot")

        if st.button("🆕 新对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("⭐ 自选股")
        favs = profile.get_favorites(user_id)
        if favs:
            for sym in favs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"{sym}", key=f"fav_{sym}", use_container_width=True):
                        st.session_state.pending_input = f"分析一下 {sym}"
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"rm_{sym}"):
                        profile.remove_favorite(user_id, sym)
                        st.rerun()
        else:
            st.caption("暂无自选股，在对话中会自动学习偏好")

        st.divider()
        qr = quota.check(user_id)
        st.metric("今日用量", f"{qr.used}/{qr.limit}", delta=f"剩余 {qr.remain} 次")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📊 大盘", use_container_width=True):
                st.session_state.pending_input = "今天大盘怎么样？"
                st.rerun()
        with c2:
            if st.button("🔥 热点", use_container_width=True):
                st.session_state.pending_input = "最近哪些板块比较热门？"
                st.rerun()
        with c3:
            if st.button("📰 快讯", use_container_width=True):
                st.session_state.pending_input = "今天有什么重要新闻？"
                st.rerun()

        st.divider()
        if st.button("🚪 退出登录", use_container_width=True):
            # Delete session from DB
            token = st.session_state.get("session_token")
            auth = st.session_state.get("auth")
            if token and auth:
                auth.delete_session(token)
            # Clear cookie via JS
            _clear_session_cookie()
            # Clear session state
            del st.session_state["user"]
            if "session_token" in st.session_state:
                del st.session_state["session_token"]
            if "messages" in st.session_state:
                del st.session_state["messages"]
            st.rerun()

    # ── 主聊天区 ──
    st.caption(f"当前用户: {user['username']}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_input" in st.session_state and st.session_state.pending_input:
        prompt = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        prompt = None

    # 新用户欢迎语
    if not st.session_state.messages:
        st.markdown("""
        ### 👋 欢迎使用 StockBot！

        我可以帮你：
        - 📈 **实时行情** — 查询股票价格、涨跌幅、成交量
        - 📊 **技术分析** — 多情景走势推演（乐观/中性/悲观）
        - 💰 **财务数据** — PE、PB、ROE、EPS 等核心指标
        - 📰 **市场热点** — 大盘概况、板块热度

        试试问我：

        """)
        examples = [
            "最近哪些板块比较热门？",
            "分析贵州茅台 600519",
            "找一些高股息银行股",
        ]
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            with cols[i]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    prompt = ex

    # 消息历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    input_text = st.chat_input("输入股票代码或问题...")

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
                        st.error(f"今日免费额度已用完（{evt.used}/{evt.limit}），明天自动重置。")
                        return
                    if isinstance(evt, TextDelta):
                        response_text += evt.content
                        response_placeholder.markdown(response_text)
                    if isinstance(evt, ToolCallStart):
                        status_placeholder.info(f"🔧 正在调用 {evt.name}...")
                    if isinstance(evt, ToolCallEnd):
                        status_placeholder.success(f"✅ {evt.name} 完成")
                    if isinstance(evt, Error):
                        st.error(f"出错了: {evt.message}")
                        return
                status_placeholder.empty()

            asyncio.run(stream())

            if response_text:
                disclaimer = "\n\n---\n> ⚠️ 以上分析仅供参考，不构成投资建议。"
                response_placeholder.markdown(response_text + disclaimer)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
