import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSION_VERSION = 1

def save_session(path: str, messages:list[dict], summary:Optional[str]):
    """
    保存对话记录
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SESSION_VERSION,
        "update_at":datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
    }

    tem_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tem_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tem_path, file_path)


def load_session(path: str)->dict:
    """
    读取会话
    """
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"⚠️  会话文件损坏，已忽略（{e}）")
        return None

    if not isinstance(payload, dict) or "messages" not in payload:
        print("⚠️  会话文件格式不识别，已忽略")
        return None

    return {
        "summary": payload.get("summary"),
        "messages": payload.get("messages", []),
    }

def delete_session(path: str) -> None:
    """
    删除会话文件，不存在时静默。
    """
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()
