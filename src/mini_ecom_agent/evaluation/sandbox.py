"""
评估沙箱：隔离、可复现地重跑测试集，并采集运行全过程（第9期）。

沙箱做三件事：
1. 隔离：每条用例独立的临时 session 文件；关闭记忆读写（否则 default.json 会注入prompt 污染评分）；
   关闭 MCP，使用本地工具。订单等模拟数据可作为幻觉检测依据，知识检索仍取决于当前 RAG 配置。
2. 插桩：单/多 Agent 都只共享一个 OpenAI client 实例，给它的chat.completions.create / beta.chat.completions.parse 打补丁，
   即可捕获整个会话所有 LLM 调用的 token、被请求的工具、延迟——无需改动 chat.py / orchestrator.py。
   工具返回值则通过包裹 ToolManager.execute_tools 采集。
3. 执行：顺序跑完用例的多轮输入，把过程与结果填进 RunTrace 返回。

关键：绝不调用 agent.close()（会触发长期记忆巩固的 LLM 写入，污染且烧钱）；
所有补丁在 finally 中还原。
"""
import tempfile
import time
from pathlib import Path

from mini_ecom_agent.config.settings import settings
from mini_ecom_agent.evaluation.dataset import EvalCase
from mini_ecom_agent.evaluation.trace import RunTrace, ToolObservation, LLMCallRecord


class Sandbox:
    """Agent 评估沙箱：构建隔离环境、插桩采集、跑用例产出 RunTrace。"""

    def __init__(self, mode:str='single',tmp_root:str|None=None):
        self.mode = mode
        self.tmp_root = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="eval_sandbox_"))
        self.tmp_root.mkdir(exist_ok=True, parents=True)

    def session_path_for(self, case_id: str) -> str:
        return str(self.tmp_root / f"{case_id}.json")

    def _build_agent(self,session_path:str):
        """在隔离配置下构建被测 Agent。"""
        # 关闭记忆读写与 MCP，保证可复现（agent.__init__ 直接读全局 settings）
        settings.memory_enabled = False
        settings.mcp_enabled = False
        if self.mode == "multi":
            from mini_ecom_agent.multi_agent.orchestrator import MultiAgentOrchestrator
            return MultiAgentOrchestrator(session_path=session_path)
        from mini_ecom_agent.agents.chat import EcomAgent
        return EcomAgent(session_path=session_path)

    def _instrument(self, agent, trace:RunTrace, patches:list[tuple]) -> None:
        """
        给共享 client、各 ToolManager、（多 Agent）Router 打补丁。

        给 Agent 的关键函数“包一层”，记录运行过程，但不修改原函数本身的核心逻辑。
        patches 保存所有被替换的原函数：
        """
        # 1) LLM client：create + beta.parse
        completions = agent.client.chat.completions                                 # 包装 LLM 调用
        patches.append((completions, "create", completions.create))                 # 取得 OpenAI 客户端的普通聊天接口。
        completions.create = self._wrap_create(completions.create, trace)           # 先保存原来的 create 函数，方便最后恢复。

        beta_completions = agent.client.beta.chat.completions                       #  包装结构化输出接口
        patches.append((beta_completions, "parse", beta_completions.parse))
        beta_completions.parse = self._wrap_parse(beta_completions.parse, trace)

        # 2) 工具执行
        for tm in self._tool_managers(agent):                                       # 包装工具执行
            patches.append((tm, "execute_tools", tm.execute_tools))
            tm.execute_tools = self._wrap_execute_tool(tm.execute_tools, trace)

        # 3) 多 Agent 路由                                                           # 包装多 Agent 路由
        if self.mode == "multi" and hasattr(agent, "router"):
            patches.append((agent.router, "router", agent.router.router))
            agent.router.router = self._wrap_route(agent.router.router, trace)

    def _tool_managers(self, agent) -> list:
        if self.mode == "multi" and hasattr(agent, "agents"):
            return [a.tools_manager for a in agent.agents.values()]
        if hasattr(agent, "tool_manager"):
            return [agent.tool_manager]
        return []

    def _close_tool_managers(self, agent) -> None:
        for tm in self._tool_managers(agent):
            try:
                tm.close() # toolmanager的关闭，主要关闭mcp连接
            except Exception:  # noqa: BLE001
                pass

    def run(self,case:EvalCase)->RunTrace:
        """跑一条用例，返回采集到的运行轨迹。"""
        trace = RunTrace(case.id,list(case.turns))
        session_path = self.session_path_for(case.id)

        agent = None
        patches: list[tuple] = []                               # (obj, attr, original) 供还原

        try:
            agent = self._build_agent(session_path)
            self._instrument(agent, trace, patches)

            result = None
            for turn in case.turns:
                result = agent.chat(turn)
            trace.final_response = result

        except Exception as e:  # noqa: BLE001 —— 单条用例异常不应中断整轮评估
            trace.error = f"{type(e).__name__}: {e}"
        finally:
            for obj, attr, original in patches:
                setattr(obj, attr, original)
            if agent is not None:
                self._close_tool_managers(agent)
            # 注意：刻意不调用 agent.close()，避免长期记忆巩固写入，是说agent.close中也有tm.close,但是还有保存session的，所以不调用agent.close

        return trace

    def _wrap_create(self, original, trace:RunTrace):
        def wrapper(*args, **kwargs):
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(
                trace, response, latency_ms,
                purpose="react",
            )
            return response

        return wrapper

    def _wrap_parse(self, original, trace:RunTrace):
        def wrapper(*args, **kwargs):
            start = time.time()
            response = original(*args, **kwargs)
            latency_ms = (time.time() - start) * 1000
            self._record_llm_call(trace, response, latency_ms, purpose="extract")
            return response
        return wrapper

    def _wrap_execute_tool(self, original, trace:RunTrace):
        def wrapper(name: str, arguments: dict) -> str:
            result_str = original(name, arguments)
            trace.tool_observations.append(
                ToolObservation(name=name, arguments=dict(arguments), result=result_str)
            )
            return result_str
        return wrapper

    def _wrap_route(self, original, trace:RunTrace):
        def wrapper(*args, **kwargs):
            first_call = len(trace.llm_calls)
            try:
                agent_key = original(*args, **kwargs)
            finally:
                for call in trace.llm_calls[first_call:]:
                    call.purpose = "router"
            trace.route = agent_key
            return agent_key
        return wrapper


    # ---------- 辅助 ----------
    @staticmethod
    def _record_llm_call(trace: RunTrace, response, latency_ms: float, purpose: str) -> None:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        tool_calls: list[dict] = []
        try:
            message = response.choices[0].message
            for tc in (getattr(message, "tool_calls", None) or []):
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        except (AttributeError, IndexError):
            pass

        model = getattr(response, "model", "") or ""
        trace.llm_calls.append(LLMCallRecord(
            purpose=purpose,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            finish_reason=getattr(response.choices[0], "finish_reason", None) if response.choices else None,
            response_text=(
                getattr(response.choices[0].message, "content", None)
                if response.choices and not tool_calls else None
            ),
        ))
