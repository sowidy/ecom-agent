import json

from mini_ecom_agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from mini_ecom_agent.tools.registry import execute_tool as local_execute_tool


def _format_connection_error(error: BaseException) -> str:
    """展开异步任务组异常，让连接日志显示实际失败原因。"""
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(_format_connection_error(exc) for exc in error.exceptions)
    return f"{type(error).__name__}: {error}"


class ToolManager:
    """聚合本地工具和 MCP 工具，提供统一的工具定义和调度接口。"""
    def __init__(self,use_mcp:bool=False, mcp_server_url:str= ""):
        self._mcp_client = None
        self._tool_source:dict[str,str]=dict()
        self._tools_defs:list[dict]=[]

        if use_mcp and mcp_server_url:
            self._init_mcp(mcp_server_url)
        else:
            self._init_local()

    def _init_local(self):
        """
        只加载本地工具
        """
        self._tools_defs = list(LOCAL_TOOL_DEFINITIONS)
        for td in self._tools_defs:
            self._tool_source[td['function']['name']] = 'local'

    def _init_mcp(self,mcp_server_url:str):
        """
        连接MCP server 加载工具，失败时降级到本地工具
        """
        from mini_ecom_agent.mcp_client import MCPClient

        client = None
        try:
            client = MCPClient(mcp_server_url)
            mcp_tools = client.connect()
            print(f"🔗 [MCP] 已连接 {mcp_server_url}，发现 {len(mcp_tools)} 个工具")

            self._mcp_client = client
            self._tools_defs = list(mcp_tools)
            self._tool_source = {
                td['function']['name']: 'mcp' for td in mcp_tools
            }

            # 只补充 MCP 没有提供的本地工具，避免同名工具重复
            for td in LOCAL_TOOL_DEFINITIONS:
                name = td['function']['name']
                if name not in self._tool_source:
                    self._tools_defs.append(td)
                    self._tool_source[name] = 'local'
        except Exception as e:
            detail = _format_connection_error(e)
            print(f"⚠️  [MCP] 连接失败 {mcp_server_url} ({detail})，降级使用本地工具")
            if client:
                client.close()
            self._mcp_client = None
            self._init_local()

    @property
    def tool_definitions(self) -> list[dict]:
        return self._tools_defs

    def execute_tools(self, name:str, arguments: dict):
        source = self._tool_source[name]
        print(f"工具来源：{source}")
        if source == 'mcp' and self._mcp_client:
            return self._mcp_client.call_tool(name, arguments)
        if source == 'local':
            return local_execute_tool(name, arguments)

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)


    def close(self):
        if self._mcp_client:
            self._mcp_client.close()
            self._mcp_client = None
