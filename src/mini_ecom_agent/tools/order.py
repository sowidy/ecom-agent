from mini_ecom_agent.tools.mock_data import ORDERS


def query_order(order_id: str) -> dict:
    """根据订单号查询订单详情，包括状态、商品、金额、物流等信息。"""
    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单 {order_id}，请核实订单号"}
    return {"success": True, "order": order}


def list_user_orders(user_name: str) -> dict:
    """按模拟数据中的用户名列出订单摘要，供用户确认具体订单。"""
    user_name = user_name.strip()
    if not user_name:
        return {"success": False, "error": "请先提供用户名"}

    orders = [
        {
            "order_id": order["order_id"],
            "status": order["status"],
            "items": [item["name"] for item in order["items"]],
            "total": order["total"],
            "created_at": order["created_at"],
        }
        for order in ORDERS.values()
        if order.get("user") == user_name
    ]
    orders.sort(key=lambda order: order["created_at"], reverse=True)
    return {
        "success": True,
        "user_name": user_name,
        "count": len(orders),
        "orders": orders,
    }
