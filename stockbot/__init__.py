import logging
from pathlib import Path
from stockbot.config import load_config
from stockbot.llm.deepseek import DeepSeekProvider
from stockbot.tools.registry import ToolRegistry
from stockbot.tools.stock_search import create_search_tool
from stockbot.tools.stock_price import create_price_tool
from stockbot.tools.stock_finance import create_finance_tool
from stockbot.tools.stock_trend import create_trend_tool
from stockbot.tools.stock_news import create_news_tool
from stockbot.tools.stock_quant import create_quant_tool
from stockbot.quant.predictor import QuantPredictor
from stockbot.data.akshare_provider import AkshareProvider
from stockbot.memory.store import MemoryStore
from stockbot.memory.history import ConversationHistory
from stockbot.memory.profile import ProfileManager
from stockbot.context import ContextAssembler
from stockbot.quota import QuotaManager
from stockbot.auth import AuthManager
from stockbot.core import AgentCore

LOGGER = logging.getLogger(__name__)


def ensure_data_dir(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_agent(config_path: str = "config.yaml", db_path: str | None = None):
    """Factory: wire up all components and return (AgentCore, MemoryStore, AuthManager, config)."""
    cfg = load_config(config_path)

    if db_path is None:
        db_path = cfg.get("memory", {}).get("db_path", "data/stockbot.db")

    ensure_data_dir(db_path)
    store = MemoryStore(db_path)
    store.init_schema()

    data_provider = AkshareProvider()

    llm_cfg = cfg.get("llm", {})
    llm = DeepSeekProvider(
        model=llm_cfg.get("model", "deepseek-chat"),
        api_key=llm_cfg.get("api_key"),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        temperature=llm_cfg.get("temperature", 0.3),
    )

    tool_registry = ToolRegistry()
    tool_registry.register(create_search_tool(data_provider))
    tool_registry.register(create_price_tool(data_provider))
    tool_registry.register(create_finance_tool(data_provider))
    tool_registry.register(create_trend_tool(data_provider))
    tool_registry.register(create_news_tool(data_provider))

    qlib_cfg = cfg.get("qlib", {})
    if qlib_cfg.get("enabled", True):
        model_dir = qlib_cfg.get("model_dir", "data/qlib_model")
        if QuantPredictor.is_available(model_dir):
            try:
                data_dir = str(Path(qlib_cfg["data_dir"]).expanduser())
                quant_predictor = QuantPredictor(
                    data_dir=data_dir,
                    model_dir=model_dir,
                    instruments=qlib_cfg.get("instruments", "csi300"),
                )
                tool_registry.register(create_quant_tool(quant_predictor))
                LOGGER.info("Qlib quant predictor loaded")
            except Exception as e:
                LOGGER.warning("Qlib quant predictor failed to load: %s", e)
        else:
            LOGGER.info("Qlib model not found at %s; run scripts/setup_qlib.py", model_dir)

    memory_cfg = cfg.get("memory", {})
    history = ConversationHistory(store, history_limit=memory_cfg.get("history_limit", 200))
    profile = ProfileManager(store)

    context_assembler = ContextAssembler(tool_registry, profile, history)

    quota_cfg = cfg.get("quota", {})
    quota = QuotaManager(store, daily_limit=quota_cfg.get("daily_limit", 5))

    auth_cfg = cfg.get("auth", {})
    auth = AuthManager(store, open_registration=auth_cfg.get("open_registration", True))

    agent = AgentCore(
        llm=llm, tool_registry=tool_registry,
        context_assembler=context_assembler, memory_store=store,
        history=history, profile=profile, quota=quota, max_turns=8,
    )

    return agent, store, auth, cfg
