from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from mini_ecom_agent.agents.memory.long_term import MemoryFact
from mini_ecom_agent.prompts.memory import LTM_EXTRACTION_PROMPT, STM_EXTRACTION_PROMPT


def _build_transcript(messages: list[dict]) -> str:
    """将消息列表格式化为文本摘要（复用 summarizer 的逻辑）。"""
    line = []
    for msg in messages:
        role = msg['role']
        content = msg['content'] or ''
        if role == 'user':
            line.append(f'用户：{content}')
        if role == 'assistant':
            tool_call = msg.get('tool_calls')
            if tool_call:
                for tc in tool_call:
                    fun = tc.get('function',

                                {})
                    name = fun.get('name',"?")
                    line.append(f'客服：[调用工具]{name}')
            if content:
                line.append(f'客服：{content}')
        elif role == 'tool':
            display = content if len(content) <= 200 else content[:200] + '...'
            line.append(f'[工具调用]：{display}')

    return '\n'.join(line)


def extract_short_term_facts(
    client: OpenAI,
    model:str,
    recent_messages: list[dict],
    existing_facts: list[str],
):
    """从最近对话中提取/更新短期记忆事实。"""
    transcript = _build_transcript(recent_messages)
    if not transcript.strip():
        return existing_facts

    existing_text = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "（暂无）"
    prompt = STM_EXTRACTION_PROMPT.format(existing_facts=existing_text)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    raw = response.choices[0].message.content.strip()

    if "无新信息" in raw:
        return existing_facts

    new_facts = [line.strip().lstrip("- ") for line in raw.splitlines() if line.strip()]
    if not new_facts:
        return existing_facts

    merged = list(existing_facts)
    existing_lower = {f.lower() for f in merged}
    for fact in new_facts:
        if fact.lower() not in existing_lower:
            merged.append(fact)
            existing_lower.add(fact.lower())
    return merged


def extract_long_term_facts(
    client: OpenAI,
    model: str,
    messages: list[dict],
    summary: str | None,
    existing_facts: list,
) -> tuple[list[MemoryFact], str]:
    """从完整会话中提取长期记忆事实 + 交互摘要。"""
    from mini_ecom_agent.agents.memory.long_term import MemoryFact
    parts = []
    if summary:
        parts.append(f"【对话摘要】\n{summary}")

    transcript = _build_transcript(messages)
    if transcript:
        parts.append(f"【对话内容】\n{transcript}")

    if not parts:
        return [], ""

    existing_text = (
        "\n".join(f"- [{f.category}] {f.content}" for f in existing_facts)
        if existing_facts
        else "（暂无）"
    )
    prompt = LTM_EXTRACTION_PROMPT.format(existing_ltm=existing_text)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], "（提取失败）"

    now = datetime.now().isoformat(timespec="seconds")
    new_facts = []
    for item in data.get("facts", []):                          # 为什么这里不用像短期比较存在性了
        content = item.get("content", "").strip()
        category = item.get("category", "other")
        if content:
            new_facts.append(MemoryFact(
                content=content,
                category=category,
                created_at=now,
            ))

    interaction_summary = data.get("interaction_summary", "")
    return new_facts, interaction_summary
