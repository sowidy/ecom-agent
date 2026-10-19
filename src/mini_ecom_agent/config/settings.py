from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    # 基础模型配置
    LLM_MODEL_NAME :str = ""
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    TEMPERATURE : float = 0.7

    # 会话管理
    session_path: str = str(BASE_DIR / "session" / "session.json")
    history_threshold:int = 10              # 消息压缩策略通常为上下文达到一定的token数，例如claude code通常为达到最大上下文窗口的70%左右，此处简略为原始消息条数超过10轮
    history_keep_recent:int = 3             # 压缩时保留最近 3 条原始消息

    # ReAct 循环
    max_react_steps: int = 5

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()