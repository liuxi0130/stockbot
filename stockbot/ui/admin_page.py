import streamlit as st
from stockbot.admin import AdminService
from stockbot.quota import QuotaManager


def render_admin(admin_svc: AdminService, quota: QuotaManager):
    st.title("Admin Panel")

    tabs = st.tabs(["Users", "Stats"])

    with tabs[0]:
        st.subheader("User List")
        users = admin_svc.list_users_with_stats()

        if users:
            for u in users:
                cols = st.columns([2, 1, 1, 1, 1, 1])
                cols[0].write(u["username"])
                cols[1].write(u["role"])
                cols[2].write(str(u.get("daily_quota", 5)))
                cols[3].write(f'{u["calls_today"]}/{u["limit"]}')
                cols[4].write(str(u["total_conversations"]))

                with cols[5].popover("Approve"):
                    extra = st.number_input("Extra quota", min_value=1, max_value=100,
                                            value=5, key=f"extra_{u['id']}")
                    if st.button("Confirm", key=f"approve_{u['id']}"):
                        admin_svc.approve_user(u["id"], extra)
                        st.success(f"Added {extra} calls for {u['username']}")
                        st.rerun()

    with tabs[1]:
        st.subheader("Global Stats")
        stats = admin_svc.get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Users", stats["total_users"])
        c2.metric("Active (7d)", stats["active_users_7d"])
        c3.metric("Calls Today", stats["total_calls_today"])
