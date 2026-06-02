from stockbot.tools.base import Tool


def create_index_predict_tool(predictor) -> Tool:
    """Create a predict_index tool backed by *predictor* (IndexPredictor instance)."""

    async def predict_index(index_code: str = "000001", period: str = "3m") -> str:
        return predictor.predict(index_code, period)

    return Tool(
        name="predict_index",
        description=(
            "预测大盘指数短期和中期走势方向。基于技术规则和可选机器学习模型，"
            "输出综合评分(0-100)和方向信号(看多/中性/看空)。"
            "支持上证指数(000001)、深证成指(399001)、创业板指(399006)。"
            "注意: ML增强模型可能需要额外配置，未配置时使用纯规则预测。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "index_code": {
                    "type": "string",
                    "description": "指数代码: 000001(上证指数), 399001(深证成指), 399006(创业板指)",
                    "default": "000001",
                },
                "period": {
                    "type": "string",
                    "description": "数据周期: 1m(1月), 3m(3月,默认), 6m(半年), 1y(1年)",
                    "default": "3m",
                },
            },
            "required": [],
        },
        func=predict_index,
    )
