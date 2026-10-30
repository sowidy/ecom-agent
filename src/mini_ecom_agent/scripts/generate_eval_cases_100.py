"""从固定的模拟商品、订单和知识文档生成可复现的 100 条评估用例。

用法：python src/mini_ecom_agent/scripts/generate_eval_cases_100.py
只生成数据，不请求模型 API。已有的 10 条基线用例不会被覆盖。
"""

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mini_ecom_agent.evaluation.dataset import EvalCase  # noqa: E402
from mini_ecom_agent.tools.mock_data import ORDERS, PRODUCTS  # noqa: E402

OUTPUT = ROOT / "src" / "mini_ecom_agent" / "evaluation" / "cases_100.json"

PRODUCT_KEYWORDS = {
    "SHOE-270-BK-42": "Nike",
    "ELEC-APP-002": "AirPods",
    "PHONE-MI14U-BK": "小米",
    "CLOTH-LEVI-501-30": "Levi",
    "HOME-DYSON-V15": "戴森",
    "ACC-AP-CASE-01": "保护壳",
}


def build_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
    counts: Counter[str] = Counter()

    def add(group: str, description: str, turns: list[str], **expected):
        counts[group] += 1
        cases.append(EvalCase(
            id=f"{group}_{counts[group]:02d}",
            description=description,
            turns=turns,
            **expected,
        ))

    # 15 条：五种订单状态及商品组合，三种明确的查询目标。
    order_questions = (
        ("当前状态", "请查一下订单 {id} 的当前状态。"),
        ("下单金额", "订单 {id} 是什么时候下单的，实付多少钱？"),
        ("商品明细", "{id} 买了什么商品？帮我核对一下。"),
    )
    for order in ORDERS.values():
        keyword = PRODUCT_KEYWORDS[order["items"][0]["sku"]]
        for subject, template in order_questions:
            add(
                "order", f"{subject}：{order['status']}",
                [template.format(id=order["order_id"])],
                expected_intent="ORDER_QUERY",
                expected_keywords=[keyword] if subject == "商品明细" else [order["order_id"]],
                expected_requires_human=False,
                expected_tools=["query_order"],
                min_tool_calls=1,
                max_tokens=15000,
                expected_route="postsale",
            )

    # 15 条：有轨迹订单要求查询物流；待发货订单查询状态即可说明无运单。
    logistics_questions = (
        "订单 {id} 的物流到哪了？",
        "帮我看一下 {id} 的快递轨迹和承运商。",
        "{id} 现在有运单号吗，预计什么时候到？",
    )
    for order in ORDERS.values():
        has_tracking = bool(order["tracking_number"])
        for template in logistics_questions:
            add(
                "logistics", f"物流查询：{order['status']}",
                [template.format(id=order["order_id"])],
                expected_intent="ORDER_QUERY",
                expected_keywords=[order["carrier"]] if has_tracking else ["发货"],
                expected_requires_human=False,
                expected_tools=["query_order", "query_logistics"] if has_tracking else ["query_order"],
                min_tool_calls=2 if has_tracking else 1,
                max_tokens=18000,
                expected_route="postsale",
            )

    # 18 条：只问数据中存在的商品，避免 query_product 随机兜底商品污染答案。
    product_questions = (
        "商品{name}还有库存吗？",
        "{name}现在卖多少钱？",
        "{name}有哪些规格和卖点？",
    )
    for product in PRODUCTS.values():
        keyword = PRODUCT_KEYWORDS[product["product_id"]]
        for template in product_questions:
            add(
                "product", f"商品咨询：{product['product_id']}",
                [template.format(name=product["name"])],
                expected_intent="PRODUCT_CONSULT",
                expected_keywords=[keyword],
                expected_requires_human=False,
                expected_tools=["query_product"],
                min_tool_calls=1,
                max_tokens=13000,
                expected_route="presale",
            )

    # 20 条：四份知识文档各取五个不同政策点，模糊意图不硬判为订单/售后。
    policy_questions = {
        "return_policy": [
            ("七天无理由退货需要满足什么条件？", "七天", "RETURN_REQUEST", "postsale"),
            ("商品有质量问题可以在多久内申请退换？", "质量", "RETURN_REQUEST", "postsale"),
            ("七天无理由退货的运费谁承担？", "运费", "RETURN_REQUEST", "postsale"),
            ("退款审核通过后多久能到账？", "退款", "RETURN_REQUEST", "postsale"),
            ("一笔订单里能只退其中一件商品吗？", "部分退款", "RETURN_REQUEST", "postsale"),
        ],
        "shipping_policy": [
            ("订单多少元可以包邮？", "99", None, None),
            ("偏远地区配送是否加收运费？", "偏远", None, None),
            ("我能指定承运快递公司吗？", "快递", None, None),
            ("普通快递送到一线城市通常要多久？", "配送", None, None),
            ("物流三天没更新该怎么处理？", "物流", None, "postsale"),
        ],
        "member_policy": [
            ("钻石会员能享受什么退货权益？", "无忧退", None, None),
            ("黄金会员有哪些包邮权益？", "包邮", None, None),
            ("消费后积分是怎么计算的？", "积分", None, None),
            ("会员等级能手动升级吗？", "升级", None, None),
            ("会员收费吗，需要交年费吗？", "免费", None, None),
        ],
        "faq_policy": [
            ("下单后一般多久发货？", "48", None, "postsale"),
            ("平台支持哪些支付方式？", "支付宝", None, None),
            ("未发货的订单还能修改收货地址吗？", "地址", None, "postsale"),
            ("商品降价后还能申请价保吗？", "价保", None, None),
            ("退款到银行卡一般多久？", "银行卡", "RETURN_REQUEST", "postsale"),
        ],
    }
    for group, questions in policy_questions.items():
        for question, keyword, intent, route in questions:
            add(
                group, "知识库政策检索", [question],
                expected_intent=intent,
                expected_keywords=[keyword],
                expected_tools=["search_knowledge"],
                min_tool_calls=1,
                max_tokens=13000,
                expected_route=route,
            )

    # 10 条：先核对订单并解释流程；不预设应调用 apply_refund。
    refund_questions = (
        "我想退订单 {id}，请先查一下现在是什么状态。",
        "订单 {id} 想申请退款，帮我确认目前进度和下一步。",
    )
    for order in ORDERS.values():
        for template in refund_questions:
            add(
                "refund", f"退款咨询：{order['status']}",
                [template.format(id=order["order_id"])],
                expected_intent="RETURN_REQUEST",
                expected_keywords=["退款"],
                expected_requires_human=False,
                expected_tools=["query_order"],
                min_tool_calls=1,
                max_tokens=15000,
                expected_route="postsale",
            )

    # 5 条：提供真实姓名，list_user_orders 才有合法必填参数。
    for order in ORDERS.values():
        add(
            "list_orders", "按已知用户名列出订单",
            [f"我叫{order['user']}，请列出我的订单。"],
            expected_intent="ORDER_QUERY",
            expected_keywords=[PRODUCT_KEYWORDS[order["items"][0]["sku"]]],
            expected_requires_human=False,
            expected_tools=["list_user_orders"],
            min_tool_calls=1,
            max_tokens=10000,
            expected_route="postsale",
        )

    # 7 条：前五条先补订单号；后两条在已查订单基础上追问物流。
    for order in ORDERS.values():
        add(
            "multi_turn", "跨轮次补充订单号",
            ["帮我看看之前下的订单。", f"订单号是 {order['order_id']}。"],
            expected_intent="ORDER_QUERY",
            expected_keywords=[PRODUCT_KEYWORDS[order["items"][0]["sku"]]],
            expected_requires_human=False,
            expected_tools=["query_order"],
            min_tool_calls=1,
            max_tokens=26000,
            expected_route="postsale",
        )
    for order_id in ("ORD-20240115-001", "ORD-20240110-003"):
        order = ORDERS[order_id]
        add(
            "multi_turn", "跨轮次追问物流",
            [f"查一下订单 {order_id}。", "再帮我看一下快递目前到哪了？"],
            expected_intent="ORDER_QUERY",
            expected_keywords=[order["carrier"]],
            expected_requires_human=False,
            expected_tools=["query_order", "query_logistics"],
            min_tool_calls=2,
            max_tokens=26000,
            expected_route="postsale",
        )

    complaints = (
        "你们的服务态度太差了，我要正式投诉并转人工处理！",
        "商品质量有问题还不给解决，我要投诉，请人工客服介入。",
        "我对这次配送服务非常不满，要求投诉并找负责人处理。",
        "你们反复推诿退款问题，我要投诉并转接人工客服。",
        "收到的商品明显破损，我很生气，要求投诉并升级处理！",
    )
    for question in complaints:
        add(
            "complaint", "明确投诉并请求升级", [question],
            expected_intent="COMPLAINT",
            expected_requires_human=True,
            max_tokens=13000,
            expected_route="complaint",
        )

    greetings = ("你好。", "早上好！", "嗨，小夕。", "在吗？", "先跟你打个招呼，您好。")
    for question in greetings:
        add(
            "greeting", "纯问候", [question],
            expected_intent="GREETING",
            expected_requires_human=False,
            min_tool_calls=0,
            max_tokens=7000,
            expected_route="postsale",
        )

    assert len(cases) == 100, counts
    assert len({case.id for case in cases}) == 100
    return cases


def main() -> None:
    cases = build_cases()
    OUTPUT.write_text(
        json.dumps({"cases": [
            {key: value for key, value in asdict(case).items() if value is not None and value != []}
            for case in cases
        ]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(cases)} 条用例：{OUTPUT}")
    print(dict(Counter(case.id.rsplit("_", 1)[0] for case in cases)))


if __name__ == "__main__":
    main()
