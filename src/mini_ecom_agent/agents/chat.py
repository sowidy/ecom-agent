import json
from typing import Optional

from openai import OpenAI

from mini_ecom_agent.agents.storage import load_session, delete_session, save_session
from mini_ecom_agent.agents.summarizer import summarize
from mini_ecom_agent.config.settings import settings
from mini_ecom_agent.prompts.customer_service import SYSTEM_PROMPT
from mini_ecom_agent.schemas.response import CustomerServiceResponse
from rich import print as rprint

OUTPUT_FORMAT_PROMPT = (
    "\n\n## 输出格式\n"
    "只输出一个 JSON 对象，不要包含 markdown 代码块或任何额外文字。"
    "JSON 必须符合以下 JSON Schema：\n"
    + json.dumps(CustomerServiceResponse.model_json_schema(), ensure_ascii=False)
)


def _strip_code_fence(text: str) -> str:
    """兼容模型偶尔返回 ```json ... ``` 的情况"""
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return text


def _parse_response(text: str) -> CustomerServiceResponse:
    """只解析第一个 JSON 对象，忽略模型偶尔追加在后面的闲聊文字"""
    text = _strip_code_fence(text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"模型未返回 JSON: {text[:200]!r}")
    payload, _ = json.JSONDecoder().raw_decode(text, start)
    return CustomerServiceResponse.model_validate(payload)


class EcomAgent:
    # """电商客服 Agent —— 第一期：基于纯 Prompt + 结构化输出"""
    # """电商客服 Agent —— 第二期：多轮对话管理（summary 压缩 + JSON 持久化）"""
    """电商客服 Agent —— 第三期：ReAct Agent + Function Calling"""

    def __init__(self,session_path:Optional[str]=None):
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL_NAME
        self.temperature = settings.TEMPERATURE
        # self.messages : list[dict] = [
        #     {"role":"system","content":SYSTEM_PROMPT + OUTPUT_FORMAT_PROMPT}
        # ]

        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent

        self.raw_messages = [] # 原始对话内容
        self.summary :Optional[str] = None
        load = load_session(self.session_path)
        if load:
            self.summary = load["summary"]
            self.raw_messages = load["messages"]


    @property
    def history_size(self)->int:
        """
        原始条数
        """
        return len(self.raw_messages)

    def _complete(self) -> str:
        """调用模型并返回 content；DeepSeek 偶发返回纯空白内容，换温度重试几次"""
        last_finish_reason = None
        for temperature in (self.temperature, 1.0, 0.5, 0.9):
            # DeepSeek 不支持 json_schema，这里用 json_object 模式 + Prompt 约束 Schema
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(),
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            last_finish_reason = choice.finish_reason
            content = (choice.message.content or "").strip()
            if content:
                return content

        raise RuntimeError(f"模型返回空内容（finish_reason={last_finish_reason}），已重试 4 次")

    def chat(self, user_input) -> CustomerServiceResponse:
        """处理用户输入：ReAct 循环 → 结构化提取 → 返回结果"""
        self.raw_messages.append({"role":"user","content":user_input})

        content = self._complete()
        parsed = _parse_response(content)
        print("parsed : ",parsed)
        # 历史里只保存干净的 JSON，避免把模型追加的闲聊文字带入后续上下文
        self.raw_messages.append({"role": "assistant", "content": parsed.model_dump_json()})
        if len(self.raw_messages) > self.history_threshold:
            self._compress_history()

        save_session(self.session_path, self.raw_messages, self.summary)
        return parsed

    def reset(self):
        """重置对话历史，并删除持久化文件"""
        self.raw_messages = []
        self.summary = None
        delete_session(self.session_path)

    def save(self):
        """
        显示保存
        """
        save_session(self.session_path, self.raw_messages, self.summary)

    def _build_messages(self):
        """
        拼装完整的LLM信息列表，system_prompt + (summary system) + 原始历史
        """
        messages: list[dict] = [
            # DeepSeek 的 json_object 模式要求 prompt 中出现 "json"，必须带上输出格式约束
            {"role": "system", "content": SYSTEM_PROMPT + OUTPUT_FORMAT_PROMPT}
        ]
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
                }
            )
        messages.extend(self._render_history())
        return messages

    def _render_history(self) -> list[dict]:
        """把历史中的 assistant JSON 还原成自然语言，便于模型理解自己说过的话"""
        rendered: list[dict] = []
        for msg in self.raw_messages:
            content = msg["content"]
            if msg["role"] == "assistant":
                try:
                    parsed = CustomerServiceResponse.model_validate_json(content)
                    content = parsed.reply
                    if parsed.follow_requestions:
                        content += f"\n{parsed.follow_requestions}"
                except Exception:
                    pass
            rendered.append({"role": msg["role"], "content": content})
        return rendered

    def _compress_history(self):
        """
        压缩除最近k条以前的信息成summary
        """
        keep = self.history_keep_recent
        old_messages = self.raw_messages[:-keep]
        recent = self.raw_messages[-keep:]

        new_summary = summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = recent
        print(
            f"\n💾 [已压缩 {len(old_messages)} 条老消息 → summary "
            f"({len(new_summary)} 字)]\n"
        )

# rprint(response) 返回

# ChatCompletion(
#     id='ca43839a-84cb-42ad-acfb-9e2d0a018960',
#     choices=[
#         Choice(
#             finish_reason='stop',
#             index=0,
#             logprobs=None,
#             message=ChatCompletionMessage(
#                 content='{"intent": "GREETING", "confidence": 0.98, "reply": "你好呀～我是并夕夕的智能客服小夕，很高兴为你服务！😊
# 我这边可以帮你查询订单物流、退换货、商品咨询、售后处理、优惠活动等，请问有什么可以帮到你的吗？", "requires_human": false, "follow_requestions":
# "请问你想咨询订单、商品还是售后相关的问题呢？方便的话可以告诉我订单号或商品名称哦～"}',
#                 refusal=None,
#                 role='assistant',
#                 annotations=None,
#                 audio=None,
#                 function_call=None,
#                 tool_calls=None,
#                 reasoning_content='The user says "你好呀" — greeting. Respond with friendly greeting, ask how to help. JSON output.\n\nNote the schema has "follow_requestions"
# (typo in schema) — optional. Provide follow_requestions as a question string or null.\n\nIntent: GREETING, confidence high.'
#             )
#         )
#     ],
#     created=1792391482,
#     model='deepseek-flash',
#     object='chat.completion',
#     metadata=None,
#     moderation=None,
#     service_tier=None,
#     system_fingerprint='e72d3c7f663d3e1383b00b3884cb4501',
#     usage=CompletionUsage(
#         completion_tokens=171,
#         prompt_tokens=711,
#         total_tokens=882,
#         completion_tokens_details=CompletionTokensDetails(
#             accepted_prediction_tokens=None,
#             audio_tokens=None,
#             reasoning_tokens=61,
#             rejected_prediction_tokens=None,
#             text_tokens=None
#         ),
#         prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cache_write_tokens=None, cached_tokens=512, image_tokens=None, text_tokens=None),
#         prompt_cache_hit_tokens=512,
#         prompt_cache_miss_tokens=199
#     )
# )