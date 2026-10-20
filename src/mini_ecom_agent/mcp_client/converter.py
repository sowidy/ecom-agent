"""MCP 工具 schema → OpenAI function calling 格式转换。"""

def mcp_tools_to_openai(mcp_tools:list):
    """
    将MCP Tool 列表转换为OpenAI function calling所需的格式

    MCP Tool 的 inputSchema 本身是JSON Schema ，和OpenAI parameters 字段一致
    只需要调整外层信封结构
    """
    result = []
    for tool in mcp_tools:
        result.append({
            "type": "function",
            "function":{
                "name":tool.name,
                "description":tool.description or "",
                "parameters":tool.input_schema,
            },
        })
    return result