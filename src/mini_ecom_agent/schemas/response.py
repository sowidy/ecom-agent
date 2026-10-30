from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class IntentType(str, Enum):

    ORDER_QUERY = "ORDER_QUERY"             # 订单查询
    RETURN_REQUEST = "RETURN_REQUEST"       # 商品退货
    PRODUCT_CONSULT = "PRODUCT_CONSULT"     # 商品咨询
    COMPLAINT = "COMPLAINT"                 # 投诉
    AFTER_SALE = "AFTER_SALE"               # 售后
    PROMOTION = "PROMOTION"                 # 促销
    ACCOUNT = "ACCOUNT"                     # 账号
    GREETING = "GREETING"                   # 闲聊
    OTHER = "OTHER"

class CustomerServiceResponse(BaseModel):
    intent: IntentType = Field(description="识别到的用户意图")
    confidence :float = Field(description="意图识别置信度 0-1", ge=0.0, le=1.0)
    reply :str = Field(description="给用户的回复内容")
    requires_human :bool = Field(description="是否需要转接人工客服",default=False)
    follow_requestions :Optional[str] = Field(description="需要向用户进一步确认的问题，如果不需要则为 null",default=None)

    @field_validator("follow_requestions", mode="before")
    @classmethod
    def normalize_follow_questions(cls, value):
        """兼容模型返回多个追问，外部接口仍保持字符串或 null。"""
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "\n".join(item.strip() for item in value if item.strip()) or None
        return value
