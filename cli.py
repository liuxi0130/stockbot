#!/usr/bin/env python
"""CLI entry point for StockBot."""
from stockbot import create_agent
from stockbot.ui.cli_ui import CLIUI


def main():
    agent, store, auth, cfg = create_agent()

    user = store.get_user("local")
    if not user:
        uid = auth.register("local", "local")
        if not uid:
            uid = store.create_user("local", "", "user")
        user_id = uid
    else:
        user_id = user["id"]

    ui = CLIUI(agent, store, agent.quota, agent.profile, user_id)
    ui.run()


if __name__ == "__main__":
    main()
