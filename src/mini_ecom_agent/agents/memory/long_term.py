import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from mini_ecom_agent.agents.memory.extraction import extract_long_term_facts


@dataclass
class MemoryFact:
    """一条长期记忆事实。"""
    content: str
    category: str  # identity / preference / behavior / issue / other
    created_at: str
    source_session: str = ""

class LongTermMemory:
    """跨会话长期记忆：持久化用户知识。"""
    def __init__(
        self,
        user_id:str="default",
        memory_dir:str="session/memory",
        max_facts=50
    ):
        self.user_id = user_id
        self.memory_dir = Path(memory_dir)
        self.max_facts = max_facts
        self.facts:list[MemoryFact] = []
        self.interaction_summaries :list[dict] = []

    @property
    def memory_path(self) -> Path:
        return self.memory_dir / f"{self.user_id}.json"

    def load(self) -> None:
        """从 JSON 文件加载用户的长期记忆。"""
        if not self.memory_path.exists():
            return
        try:
            with open(self.memory_path, "r",encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for item in data.get('facts', []):
            self.facts.append(MemoryFact(
                content=item["content"],
                category=item.get("category", "other"),
                created_at=item.get("created_at", ""),
                source_session=item.get("source_session", ""),
            ))
        self.interaction_summaries = data.get('interaction_summaries', [])

    def save(self) -> None:
        """持久化到 JSON 文件（原子写入）。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        pay_load = {
            "version":1,
            "user_id": self.user_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "facts":[asdict(f) for f in self.facts],
            "interaction_summaries": self.interaction_summaries
        }

        tmp_path = self.memory_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(pay_load, f,ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.memory_path)

    def add_facts(self, new_facts:list[MemoryFact]):
        """添加新事实，自动去重并裁剪到 max_facts。"""
        existing_content = {f.content.lower() for f in self.facts}
        for fact in new_facts:
            if fact.content.lower() not in existing_content:
                self.facts.append(fact)
                existing_content.add(fact.content.lower())
        if len(self.facts) > self.max_facts:
            self.facts = self.facts[-self.max_facts:]

    def add_interaction_summaries(self, summary:str):
        self.interaction_summaries.append({
            "summary":summary,
            "timestamp":datetime.now().isoformat(timespec="seconds"),
        })

    def extract_and_save(
        self,
        client:OpenAI,
        model:str,
        messages:list[dict],
        summary:Optional[str],
    ):
        """从会话消息中提取长期记忆事实并保存。"""
        if not messages and not summary:
            return

        new_fact, interaction_summary = extract_long_term_facts(client, model, messages, summary,self.facts)
        if new_fact:
            self.add_facts(new_fact)
        if interaction_summary:
            self.add_interaction_summaries(interaction_summary)

        self.save()

    def build_prompt_section(self):
        """生成注入 system prompt 的长期记忆片段。"""
        if not self.facts and not self.interaction_summaries:
            return

        parts = []
        if self.facts:
            facts_text = '\n'.join(f"- [{f.category}] {f.content}" for f in self.facts)
            parts.append(facts_text)
        if self.interaction_summaries:
            recent = self.interaction_summaries[-3:]
            summaries_text = '\n'.join(f"- {s['summary']}" for s in recent)
            parts.append(f"最近的交互记录：\n{summaries_text}")
        return '\n\n'.join(parts)

    def reset(self):
        """清空该用户的长期记忆（文件也删除）。"""
        self.facts = []
        self.interaction_summaries = []
        if self.memory_path.exists():
            self.memory_path.unlink()
