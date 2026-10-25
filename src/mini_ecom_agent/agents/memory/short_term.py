from openai import OpenAI

from mini_ecom_agent.agents.memory.extraction import extract_short_term_facts


class ShortTermMemory:
    """会话内短期记忆：从对话中提取结构化事实，增强上下文感知。"""
    def __init__(self):
        self.facts: list[str] = []

    def update(
        self,
        client:OpenAI,
        model:str,
        recent_messages:list[dict],
    ):
        """从最近的对话消息中提取/更新事实。"""
        self.facts = extract_short_term_facts(client,model, recent_messages,self.facts)

    def build_prompt_section(self):
        """生成注入 system prompt 的短期记忆片段。"""
        if not self.facts:
            return None

        facts_text = '\n'.join(f'- {f}' for f in self.facts)
        return f"以下是本次对话中提取的用户关键信息（短期记忆）：\n{facts_text}"

    def reset(self):
        self.facts = []

    def to_dict(self):
        return {"facts": self.facts}

    @classmethod
    def from_dict(cls, data:dict):
        stm = ShortTermMemory()
        stm.facts = data.get("facts", [])
        return stm
