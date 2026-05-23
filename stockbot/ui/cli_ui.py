import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from stockbot.core import AgentCore
from stockbot.memory.store import MemoryStore
from stockbot.quota import QuotaManager
from stockbot.memory.profile import ProfileManager
from stockbot.events import TextDelta, TextDone, ToolCallStart, ToolCallEnd, Error, QuotaExceeded


WELCOME = """[bold cyan]StockBot[/] -- A-share AI Analysis Assistant (v0.1.0)

Type your question to start, /help for commands, /exit to quit"""

HELP_TEXT = """[bold]Commands:[/]
  /help       Show this help
  /tools      List all tools
  /watch CODE Add stock to watchlist
  /portfolio  View watchlist
  /quota      Check daily usage
  /clear      Clear session context
  /admin approve N  Approve extra quota
  /exit       Quit"""

PROMPT_CHAR = "> "


class CLIUI:
    def __init__(self, agent: AgentCore, store: MemoryStore,
                 quota: QuotaManager, profile: ProfileManager, user_id: str):
        self.agent = agent
        self.store = store
        self.quota = quota
        self.profile = profile
        self.user_id = user_id
        self.console = Console()
        self.running = True

    def run(self):
        self.console.print(Panel.fit(WELCOME, border_style="cyan"))
        self.console.print()

        while self.running:
            try:
                user_input = self.console.input(f"[bold white]{PROMPT_CHAR}[/] ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\nGoodbye!")
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            asyncio.run(self._chat(user_input))

    def _handle_command(self, cmd: str):
        parts = cmd.split()
        op = parts[0].lower()

        if op == "/exit":
            self.running = False
            self.console.print("Goodbye!")
        elif op == "/help":
            self.console.print(HELP_TEXT)
        elif op == "/tools":
            self.console.print(self.agent.tool_registry.describe())
        elif op == "/watch" and len(parts) > 1:
            self.profile.add_favorite(self.user_id, parts[1])
            self.console.print(f"[green]Added {parts[1]} to watchlist[/]")
        elif op == "/portfolio":
            favs = self.profile.get_favorites(self.user_id)
            if favs:
                self.console.print(f"[bold]Watchlist:[/] {', '.join(favs)}")
            else:
                self.console.print("No watchlist. Use /watch CODE to add")
        elif op == "/quota":
            qr = self.quota.check(self.user_id)
            self.console.print(
                f"[bold]Daily usage:[/] {qr.used}/{qr.limit}  (remaining {qr.remain})"
            )
        elif op == "/clear":
            self.console.print("[yellow]Session context cleared[/]")
        elif op == "/admin":
            self._handle_admin(parts)
        else:
            self.console.print(f"[red]Unknown command: {op}[/]")

    def _handle_admin(self, parts: list[str]):
        if len(parts) < 2:
            self.console.print("Usage: /admin approve <n>")
            return
        sub = parts[1].lower()
        if sub == "approve" and len(parts) > 2:
            try:
                n = int(parts[2])
                self.quota.approve(self.user_id, n)
                self.console.print(f"[green]Approved {n} extra calls[/]")
            except ValueError:
                self.console.print("[red]Invalid number[/]")
        else:
            self.console.print("Usage: /admin approve <n>")

    async def _chat(self, user_input: str):
        response_text = ""

        async for evt in self.agent.run(self.user_id, user_input):
            if isinstance(evt, QuotaExceeded):
                self.console.print(
                    f"\n[red]Daily quota exhausted ({evt.used}/{evt.limit})[/]\n"
                    "Contact admin for more quota, or wait until tomorrow."
                )
                return

            if isinstance(evt, TextDelta):
                response_text += evt.content

            if isinstance(evt, TextDone):
                self.console.print()
                self.console.print(Markdown(response_text))
                qr = self.quota.check(self.user_id)
                self.console.print(f"[dim]Used {qr.used}/{qr.limit} today[/]")

            if isinstance(evt, ToolCallStart):
                self.console.print(f"  [yellow]🔧 {evt.name}(...)[/]", end="")

            if isinstance(evt, ToolCallEnd):
                self.console.print(f" [green]✓[/]")

            if isinstance(evt, Error):
                self.console.print(f"\n[red]✗ {evt.message}[/]")
                return

        self.console.print()
