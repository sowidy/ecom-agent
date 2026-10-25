from typing import Optional

from openai import OpenAI

from mini_ecom_agent.agents.memory.long_term import LongTermMemory
from mini_ecom_agent.agents.memory.short_term import ShortTermMemory


class MemoryManager:
    """记忆管理器：统一管理短期记忆和长期记忆。"""
    def __init__(
        self,
        model:str,
        client:OpenAI,
        user_id :str ='default',
        memory_dir: str = "app/sessions/memory",
        memory_enabled:bool=True,
        max_facts_limit:int=50
    ):
        self.client = client
        self.model = model
        self.memory_enabled = memory_enabled

        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory(
            user_id=user_id,
            memory_dir=memory_dir,
            max_facts=max_facts_limit,
        )

        if self.memory_enabled:
            self.ltm.load()

    def update_short_term(self, recent_messages:list[dict]):
        """每轮对话后更新短期记忆。"""
        if not self.memory_enabled:
            return
        self.stm.update(self.client,self.model,recent_messages)

    def build_memory_prompt_sections(self):
        """生成所有记忆相关的 system prompt 消息列表。"""
        if not self.memory_enabled:
            return []

        section = []
        ltm_section = self.ltm.build_prompt_section()
        if ltm_section:
            section.append({'role':'system','content':ltm_section})
        stm_section = self.stm.build_prompt_section()
        if stm_section:
            section.append({'role':'system','content':stm_section})
        return section

    def consolidate_to_ltm(
        self,
        messages:list[dict],
        summary:Optional[str]
    ):
        """会话结束时，将本次对话的关键事实巩固到长期记忆。"""
        if not self.memory_enabled:
            return
        self.ltm.extract_and_save(self.client,self.model,messages,summary)

    def reset_stm(self):
        """重置短期记忆（会话内重置时调用）。"""
        self.stm.reset()

    def reset_all(self):
        """重置所有记忆（短期+长期）。"""
        self.stm.reset()
        self.ltm.reset()

    def stm_to_dict(self):
        return self.stm.to_dict()

    def restore_stm(self, data:dict):
        self.stm = ShortTermMemory.from_dict(data)
