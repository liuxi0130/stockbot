from stockbot.tools.base import Tool
from stockbot.quant.predictor import QuantPredictor


def create_quant_tool(predictor: QuantPredictor) -> Tool:
    """Create a quant prediction tool backed by *predictor*."""

    async def predict_stock(symbol: str) -> str:
        result = predictor.predict(symbol)
        if "error" in result:
            return f"量化预测失败: {result['error']}"

        score = result["score"]
        rank_pct = result["rank_pct"]
        total = result["total"]
        refreshed = result["refreshed_at"]
        signal = result["signal"]
        rank_pos = int(rank_pct * total / 100)

        return (
            f"🔮 {result['symbol']} 量化因子预测\n\n"
            f"预测信号: {signal}\n"
            f"预期收益得分: {score:+.2%}\n"
            f"全市场排名: Top {rank_pct:.0f}% ({rank_pos}/{total})\n"
            f"覆盖范围: CSI300 成分股 ({total}只)\n"
            f"模型刷新: {refreshed}\n"
            f"模型: LightGBM + Alpha158因子\n\n"
            f"⚠️ 量化模型基于历史Alpha因子训练，预测结果仅供参考，"
            f"不构成投资建议。"
        )

    return Tool(
        name="predict_stock",
        description=(
            "使用量化因子模型(Alpha158+LightGBM)预测股票未来收益。"
            "返回预期收益得分、全市场排名和信号方向。"
            "仅支持CSI300成分股，适合中长期投资参考。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如 600519",
                }
            },
            "required": ["symbol"],
        },
        func=predict_stock,
    )
