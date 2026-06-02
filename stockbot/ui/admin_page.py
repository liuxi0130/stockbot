import streamlit as st
from stockbot.admin import AdminService
from stockbot.quota import QuotaManager


def render_admin(admin_svc: AdminService, quota: QuotaManager):
    st.title("🔧 管理面板")

    tabs = st.tabs(["用户管理", "数据统计", "登录记录", "搜索记录"])

    # ── Tab 1: 用户管理 ──
    with tabs[0]:
        st.subheader("用户列表")
        users = admin_svc.list_users_with_stats()

        if users:
            h_cols = st.columns([2, 1, 1, 1, 1, 1])
            h_cols[0].markdown("**用户名**")
            h_cols[1].markdown("**角色**")
            h_cols[2].markdown("**日配额**")
            h_cols[3].markdown("**今日用量**")
            h_cols[4].markdown("**总对话**")
            h_cols[5].markdown("**操作**")
            st.divider()

            for u in users:
                cols = st.columns([2, 1, 1, 1, 1, 1])
                cols[0].write(u["username"])
                cols[1].write(u["role"])
                cols[2].write(str(u.get("daily_quota", 5)))
                cols[3].write(f'{u["calls_today"]}/{u["limit"]}')
                cols[4].write(str(u["total_conversations"]))

                with cols[5].popover("提额"):
                    extra = st.number_input("新增额度", min_value=1, max_value=100,
                                            value=5, key=f"extra_{u['id']}")
                    if st.button("确认", key=f"approve_{u['id']}"):
                        admin_svc.approve_user(u["id"], extra)
                        st.success(f"已为 {u['username']} 增加 {extra} 次调用")
                        st.rerun()

    # ── Tab 2: 数据统计 ──
    with tabs[1]:
        st.subheader("全局统计")
        stats = admin_svc.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("总用户数", stats["total_users"])
        c2.metric("7 天活跃", stats["active_users_7d"])
        c3.metric("今日调用", stats["total_calls_today"])

    # ── Tab 3: 登录记录 ──
    with tabs[2]:
        st.subheader("最近登录记录")
        activities = admin_svc.get_login_history(limit=100)
        if activities:
            # Filter login events
            logins = [a for a in activities if a.get("action") == "login"]
            if logins:
                for a in logins[:30]:
                    st.text(f"{a['created_at']}  |  {a['username']}  |  {a['detail']}")
            else:
                st.caption("暂无登录记录")
        else:
            st.caption("暂无活动记录")

    # ── Tab 4: 搜索记录 ──
    with tabs[3]:
        st.subheader("最近用户问题与工具调用")

        # User queries
        st.markdown("#### 用户提问")
        queries = [a for a in admin_svc.get_login_history(limit=200)
                   if a.get("action") == "query"]
        if queries:
            for q in queries[:30]:
                st.text(f"{q['created_at']}  |  {q['username']}")
                st.caption(f"  {q['detail'][:150]}")
        else:
            st.caption("暂无搜索记录")

        st.divider()

        # Tool calls
        st.markdown("#### 工具调用")
        tool_activities = [a for a in admin_svc.get_login_history(limit=200)
                          if a.get("action") not in ("login", "query")]
        if tool_activities:
            for a in tool_activities[:30]:
                st.text(f"{a['created_at']}  |  {a['username']}  |  {a['action']}")
                if a.get("detail"):
                    st.caption(f"  {a['detail'][:100]}")
        else:
            st.caption("暂无工具调用记录")
