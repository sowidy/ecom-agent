from typing import Optional

from openai import OpenAI
from mini_ecom_agent.prompts.agents import ROUTER_PROMPT

VALID_AGENTS = {"presale", "postsale", "complaint"}
DEFAULT_AGENT = "postsale"

class Router:
    """使用 LLM 对用户意图分类，路由到对应的子 Agent。"""

    def __init__(self, client: OpenAI, model:str, max_tokens: int = 1024) -> None:
        self.client = client
        self.model = model
        if max_tokens <= 0:
            raise ValueError("路由 token 预算必须大于 0")
        self.max_tokens = max_tokens

    def router(self, user_input: str, history: Optional[list[dict]])-> str:
        """
        返回子agent标志
        """
        recent_context = ""
        if history:
            recent = [m for m in history[-4:] if m.get('role') in ('user', 'assistant')]
            if recent:
                lines = []
                for m in recent:
                    role = "用户" if m.get('role') == 'user' else "客服"
                    content = m['content']
                    if content and len(content) < 200:
                        lines.append(f'{role}: {content}')
                if lines:
                    recent_context = "\n最近对话\n" + "\n".join(lines) + "\n\n"

        prompt = ROUTER_PROMPT.format(user_input=user_input)
        if recent_context:
            prompt = recent_context + "\n" + prompt

        for attempt in range(2):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens * (2 ** attempt),
            )
            choice = response.choices[0]
            raw = (choice.message.content or "").strip().lower()
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                continue
            if raw in VALID_AGENTS:
                return raw
            raise ValueError(
                f"路由模型未返回有效分类: content={raw!r}, finish_reason={finish_reason!r}"
            )
        raise ValueError("路由输出连续被截断，请增大 ROUTER_MAX_TOKENS")



