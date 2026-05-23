# StockBot — A-Share AI Analysis Assistant

AI-powered stock analysis agent powered by DeepSeek API.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your DeepSeek API key:
```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "your-key"

# Linux / Mac
export DEEPSEEK_API_KEY="your-key"
```

3. Run CLI mode:
```bash
python cli.py
```

4. Run Web mode (multi-user):
```bash
streamlit run app.py
```

## Features

- Real-time A-share stock quotes (via akshare)
- Multi-scenario technical analysis (MA / MACD / RSI) with bullish/neutral/bearish outlooks
- Financial data lookup (PE, PB, ROE, EPS, revenue growth)
- Stock news search
- Personalized watchlist and user profiling
- Multi-user Web interface with registration/login
- Daily quota management with admin approval
- CLI mode with Rich terminal UI

## Configuration

Edit `config.yaml` to customize:
- LLM model and parameters
- Daily quota limits
- Data providers
- UI defaults

## Deployment

| Stage | Approach | Command |
|-------|----------|---------|
| Dev | Local machine | `streamlit run app.py` |
| Test | Hugging Face Spaces | Push to Space |
| Production | Cloud server + Nginx | systemd service |

## Project Structure

```
stockbot/
├── cli.py                  # CLI entry point
├── app.py                  # Web entry point (Streamlit)
├── config.yaml             # Global configuration
├── stockbot/
│   ├── __init__.py          # Factory: create_agent()
│   ├── core.py              # AgentCore — ReAct loop
│   ├── context.py           # ContextAssembler — message assembly
│   ├── quota.py             # QuotaManager — rate limiting
│   ├── auth.py              # AuthManager — register/login
│   ├── admin.py             # AdminService — user management
│   ├── events.py            # StreamEvent dataclasses
│   ├── config.py            # YAML config loader
│   ├── llm/                 # LLM providers (DeepSeek + abstract)
│   ├── tools/               # 5 stock analysis tools
│   ├── data/                # Data providers (akshare + abstract)
│   ├── memory/              # SQLite storage + history + profile
│   └── ui/                  # CLI (Rich) + Web (Streamlit) interfaces
└── tests/                   # 66 unit tests
```

## Disclaimer

All analysis is for reference only. Not investment advice.
