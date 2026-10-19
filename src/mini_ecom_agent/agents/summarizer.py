from typing import Optional

from openai import OpenAI

from mini_ecom_agent.prompts.customer_service import SYSTEM_PROMPT


def summarize(client: OpenAI, model:str, old_messages:list[dict], prev_summary:Optional[str]):
    """
    把过长的历史对话压缩成一段简短摘要，减少后续请求的 Token 消耗，同时保留上下文
    """

    parts : list[str] = []
    if prev_summary is not None:
        parts.append(f"【此前摘要】\n {prev_summary}")

    transcript_lines = []
    for msg in old_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            transcript_lines.append(f"用户：{content}")
        elif role == "assistant":
            # transcript_lines.append(f"客服：{content}")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    transcript_lines.append(f"客服：[调用工具 {name}({args})]")
            if content:
                transcript_lines.append(f"客服：{content}")
        elif role == "tool":
            display = content if len(content) <= 200 else content[:200] + "..."
            transcript_lines.append(f"[工具结果] {display}")
    parts.append("【待压缩对话】\n" + "\n".join(transcript_lines))

    user_content = "\n".join(parts)

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
    )
    return response.choices[0].message.content.strip()