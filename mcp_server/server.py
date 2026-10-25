import json
import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mini_ecom_agent.tools.order import query_order as _query_order
from mini_ecom_agent.tools.order import list_user_orders as _list_user_orders
from mini_ecom_agent.tools.product import query_product as _query_product
from mini_ecom_agent.tools.logistics import query_logistics as _query_logistics
from mini_ecom_agent.tools.refund import apply_refund as _apply_refund
from mini_ecom_agent.tools.knowledge import search_knowledge as _search_knowledge

ROOT = Path(__file__).parent

sys.path.insert(0, str(ROOT))



mcp = MCPServer("ecom-tools")

@mcp.tool()
def query_order(order_id:str):
    """根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等"""
    result = _query_order(order_id)
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def list_user_orders(user_name: str) -> str:
    """按用户名查询该用户的订单摘要。用户未提供订单号时，先确认用户名。"""
    result = _list_user_orders(user_name)
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def query_product(keyword: str) -> str:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。支持模糊搜索"""
    result = _query_product(keyword)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流轨迹信息，包括快递公司、运单号、运输状态和轨迹事件"""
    result = _query_logistics(order_id)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def apply_refund(order_id: str, reason: str) -> str:
    """为指定订单申请退款。注意：这是一个敏感操作，调用前应先与用户确认"""
    result = _apply_refund(order_id, reason)
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索并夕夕的政策与帮助文档（退换货政策、配送说明、会员权益、FAQ）。
    当顾客询问规则、流程、时效、是否支持等政策类问题时使用。
    返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策。
    """
    result = _search_knowledge(query, top_k=top_k)
    return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=9123,
    )

# 定义mcp工具后，需要工具信息，mcp会自动生成
#   {
#     "name": "query_order",
#     "description": "根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等",
#     "inputSchema": {
#       "properties": {
#         "order_id": {
#           "title": "Order Id",
#           "type": "string"
#         }
#       },
#       "required": [
#         "order_id"
#       ],
#       "type": "object",
#       "title": "query_orderArguments"
#     }
#   },
