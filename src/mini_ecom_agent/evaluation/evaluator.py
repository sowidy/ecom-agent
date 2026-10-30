"""
评估执行器：编排 沙箱跑用例 → 双层评分 → 聚合报告（第9期）。

Evaluator 自己不碰 Agent，只负责：让 Sandbox 跑出 RunTrace，再用 metrics 对轨迹打分
（过程层 + 结果层，代码规则恒算、LLM judge 受 use_judge 控制），最后聚合。
"""
from dataclasses import dataclass, field

from openai import OpenAI

from mini_ecom_agent.evaluation import metrics
from mini_ecom_agent.evaluation.dataset import EvalCase
from mini_ecom_agent.evaluation.sandbox import Sandbox


@dataclass
class EvalResult:
    """单条用例的评估结果。

    规则指标在用例提供相应预期值时评分；Judge 指标还要求启用 use_judge。
    None 表示未评分，可能因为缺少预期值、关闭 Judge、没有可计算的调用，
    或运行/评分提前失败；聚合平均分会跳过 None，不能将其当作 0 分。
    """

    case_id: str
    description: str

    # ---------- 过程指标 ----------
    tool_accuracy: float | None = None          # 规则分：有 expected_tools 时，计算预期工具命中比例；否则 None
    tool_efficiency: float | None = None        # 规则分：有 min_tool_calls 时计算；应调但实际 0 次时也为 None
    token_cost: int = 0                              # 已采集的 Agent token 数；即使运行异常也计入，0 可能是没有采集到用量
    token_pass: bool | None = None                   # 有 max_tokens 时检查是否超预算；未设置预算时为 None
    process_soundness: float | None = None      # Judge 分：工具顺序合理性 1～5 分除以 5；关闭 Judge 或评分失败时可为 None
    route_match: float | None = None            # 规则分：有 expected_route 时比较实际路由；否则 None

    # ---------- 结果指标 ----------
    intent_match: float | None = None           # 规则分：有 expected_intent 时比较最终意图；否则 None
    keyword_coverage: float | None = None       # 规则分：有 expected_keywords 时计算回复命中比例；否则 None
    requires_human_match: float | None = None   # 规则分：预期为 True/False 均评分；仅预期为 None 时跳过
    answer_quality: float | None = None         # Judge 分：回答质量 1～5 分除以 5；关闭 Judge 或评分失败时可为 None
    faithfulness: float | None = None           # Judge 分：忠实 1/不忠实 0；关闭 Judge 或提前失败时为 None，解析失败记 0

    # ---------- 汇总 ----------
    process_score: float | None = None          # 已评分过程指标的平均值；均未评分或提前失败时为 None
    result_score: float | None = None           # 已评分结果指标的平均值；均未评分或提前失败时为 None
    passed: bool = False                             # 所有已评分维度达阈值且 token 未超预算才为 True；异常时保持 False

    trace: dict = field(default_factory=dict)           # RunTrace 精简快照
    judge_reasons: dict = field(default_factory=dict)   # judge 文字理由
    error: str | None = None                            # 运行或评分异常描述；正常时为 None

def _avg(values: list[float | None]) -> float | None:
    """对非 None 项求均值；全为 None 返回 None。"""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None

class Evaluator:
    """评估器：对一批用例执行沙箱运行 + 双层评分 + 聚合。"""

    def __init__(
        self,
        sandbox: Sandbox,
        client: OpenAI,
        model: str,
        use_judge: bool = True,
        pass_threshold: float = 0.6,
    ):
        self.sandbox = sandbox
        self.client = client
        self.model = model
        self.use_judge = use_judge
        self.pass_threshold = pass_threshold

    def run_case(self, case: EvalCase) -> EvalResult:
        trace = self.sandbox.run(case)  # 得到实际运行的记录
        res = EvalResult(case_id=case.id, description=case.description, trace=trace.to_dict())
        # 已发生的调用不会因后续校验失败而免费，异常用例也必须计入消耗。
        res.token_cost = trace.total_tokens
        res.token_pass = metrics.token_cost_pass(trace.total_tokens, case.max_tokens)

        if trace.error:
            res.error = trace.error
            return res

        try:
            resp = trace.final_response
            reply = resp.reply if resp else ""
            actual_intent = resp.intent.value if resp else ""
            actual_human = resp.requires_human if resp else False
            last_input = case.turns[-1]
            called = trace.tool_call_names

            # ---------- 过程指标（代码规则）----------
            res.tool_accuracy = metrics.tool_accuracy(called, case.expected_tools)
            res.tool_efficiency = metrics.tool_efficiency(trace.num_tool_calls, case.min_tool_calls)
            res.token_pass = metrics.token_cost_pass(trace.total_tokens, case.max_tokens)
            res.route_match = metrics.route_match(trace.route, case.expected_route)

            # ---------- 结果指标（代码规则）----------
            res.intent_match = metrics.intent_match(actual_intent, case.expected_intent)
            res.keyword_coverage = metrics.keyword_coverage(reply, case.expected_keywords)
            res.requires_human_match = metrics.requires_human_match(actual_human, case.expected_requires_human)

            # ---------- LLM judge ----------
            if self.use_judge:
                q_score, q_reason = metrics.judge_answer_quality(self.client, self.model, last_input, reply, case.expected_keywords)
                res.answer_quality = q_score / 5.0 if q_score else None
                res.judge_reasons["answer_quality"] = q_reason

                f_score, f_reason = metrics.judge_faithfulness(self.client, self.model, reply, trace.tool_observations)
                res.faithfulness = f_score
                res.judge_reasons["faithfulness"] = f_reason

                p_score, p_reason = metrics.judge_process_soundness(self.client, self.model, last_input, called)
                res.process_soundness = p_score / 5.0 if p_score else None
                res.judge_reasons["process_soundness"] = p_reason

            # ---------- 汇总 ----------
            res.process_score = _avg([
                res.tool_accuracy,
                res.tool_efficiency,
                res.process_soundness,
                res.route_match,
            ])
            res.result_score = _avg([
                res.intent_match,
                res.keyword_coverage,
                res.requires_human_match,
                res.answer_quality,
                res.faithfulness,
            ])
            res.passed = self._decide_pass(res)

        except Exception as e:  # noqa: BLE001 —— 评分异常隔离到单条用例
            res.error = f"评分异常 {type(e).__name__}: {e}"

        return res

    def run_all(self, cases: list[EvalCase]) -> dict:
        results = [self.run_case(c) for c in cases]
        return self._aggregate(results)

    def _decide_pass(self, res: EvalResult) -> bool:
        """所有被指定的评分维度都 ≥ 阈值，且 token 不超预算，才算通过。"""
        if res.error:
            return False
        if res.token_pass is False:
            return False
        dims = [
            res.tool_accuracy, res.tool_efficiency, res.process_soundness, res.route_match,
            res.intent_match, res.keyword_coverage, res.requires_human_match,
            res.answer_quality, res.faithfulness,
        ]
        present = [d for d in dims if d is not None]
        return all(d >= self.pass_threshold for d in present) if present else True

    def _aggregate(self, results: list[EvalResult]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        total_tokens = sum(r.token_cost for r in results)
        summary = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "error_cases": sum(r.error is not None for r in results),
            "scored_cases": sum(r.error is None for r in results),
            "avg_process_score": _avg([r.process_score for r in results]),
            "avg_result_score": _avg([r.result_score for r in results]),
            "total_tokens": total_tokens,
            "avg_tokens_per_case": total_tokens / total if total else 0,
        }
        return {
            "summary": summary,
            "cases": [self._result_to_dict(r) for r in results],
        }

    @staticmethod
    def _result_to_dict(r: EvalResult) -> dict:
        return {
            "case_id": r.case_id,
            "description": r.description,
            "passed": r.passed,
            "process": {
                "tool_accuracy": r.tool_accuracy,
                "tool_efficiency": r.tool_efficiency,
                "token_cost": r.token_cost,
                "token_pass": r.token_pass,
                "process_soundness": r.process_soundness,
                "route_match": r.route_match,
                "process_score": r.process_score,
            },
            "result": {
                "intent_match": r.intent_match,
                "keyword_coverage": r.keyword_coverage,
                "requires_human_match": r.requires_human_match,
                "answer_quality": r.answer_quality,
                "faithfulness": r.faithfulness,
                "result_score": r.result_score,
            },
            "judge_reasons": r.judge_reasons,
            "trace": r.trace,
            "error": r.error,
        }
