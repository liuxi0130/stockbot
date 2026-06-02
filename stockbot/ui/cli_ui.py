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


WELCOME = """[bold cyan]StockBot[/] — A 股智能投资分析助手 (v0.1.0)

输入问题开始分析，/help 查看命令，/exit 退出"""

HELP_TEXT = """[bold]命令列表:[/]
  /help         显示帮助
  /tools        列出所有工具
  /watch 代码    添加自选股
  /portfolio    查看自选股
  /quota        查看今日用量
  /clear        清除对话上下文
  /admin approve N  审批额外额度
  /exit         退出"""

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
                self.console.print("\n再见！")
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
            self.console.print("再见！")
        elif op == "/help":
            self.console.print(HELP_TEXT)
        elif op == "/tools":
            self.console.print(self.agent.tool_registry.describe())
        elif op == "/watch" and len(parts) > 1:
            self.profile.add_favorite(self.user_id, parts[1])
            self.console.print(f"[green]已将 {parts[1]} 加入自选股[/]")
        elif op == "/portfolio":
            favs = self.profile.get_favorites(self.user_id)
            if favs:
                self.console.print(f"[bold]自选股:[/] {', '.join(favs)}")
            else:
                self.console.print("暂无自选股。使用 /watch 代码 添加")
        elif op == "/quota":
            qr = self.quota.check(self.user_id)
            self.console.print(
                f"[bold]今日用量:[/] {qr.used}/{qr.limit}  (剩余 {qr.remain})"
            )
        elif op == "/clear":
            self.console.print("[yellow]对话上下文已清除[/]")
        elif op == "/admin":
            self._handle_admin(parts)
        else:
            self.console.print(f"[red]未知命令: {op}[/]")

    def _handle_admin(self, parts: list[str]):
        if len(parts) < 2:
            self.console.print("用法: /admin approve <次数>")
            return
        sub = parts[1].lower()
        if sub == "approve" and len(parts) > 2:
            try:
                n = int(parts[2])
                self.quota.approve(self.user_id, n)
                self.console.print(f"[green]已审批 {n} 次额外调用[/]")
            except ValueError:
                self.console.print("[red]请输入有效数字[/]")
        else:
            self.console.print("用法: /admin approve <次数>")

    async def _chat(self, user_input: str):
        response_text = ""

        async for evt in self.agent.run(self.user_id, user_input):
            if isinstance(evt, QuotaExceeded):
                self.console.print(
                    f"\n[red]今日免费额度已用完 ({evt.used}/{evt.limit})[/]\n"
                    "请联系管理员提额，或等待明天自动重置。"
                )
                return

            if isinstance(evt, TextDelta):
                response_text += evt.content

            if isinstance(evt, TextDone):
                self.console.print()
                self.console.print(Markdown(response_text))
                qr = self.quota.check(self.user_id)
                self.console.print(f"[dim]今日已用 {qr.used}/{qr.limit} 次[/]")

            if isinstance(evt, ToolCallStart):
                self.console.print(f"  [yellow]🔧 {evt.name}(...)[/]", end="")

            if isinstance(evt, ToolCallEnd):
                self.console.print(f" [green]✓[/]")

            if isinstance(evt, Error):
                self.console.print(f"\n[red]✗ {evt.message}[/]")
                return

        self.console.print()
