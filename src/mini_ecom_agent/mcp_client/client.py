import asyncio
import json
import threading
from contextlib import suppress

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mini_ecom_agent.mcp_client.converter import mcp_tools_to_openai


class MCPClient:
    """同步 MCP 客户端：后台线程用 asyncio.run 托管事件循环，主线程同步调用。"""

    def __init__(self, server_url: str, timeout: float = 30.0):
        self._server_url = server_url
        self._timeout_seconds = timeout

        self._background_loop: asyncio.AbstractEventLoop | None = None
        self._background_thread: threading.Thread | None = None
        self._client_session: ClientSession | None = None

        # 后台线程完成初始化时置位，无论初始化成功还是失败。
        self._startup_finished = threading.Event()
        # 主线程通过该事件通知后台协程退出并关闭 MCP 会话。
        self._shutdown_requested = asyncio.Event()
        self._startup_exception: BaseException | None = None
        self._openai_tool_definitions: list[dict] = []

    def connect(self) -> list[dict]:
        """
        连接 MCP Server，发现工具，返回 OpenAI 格式的工具定义列表。
        """
        if self._background_thread is not None:
            raise RuntimeError("MCPClient 已连接，请先调用 close()")
        self._startup_exception = None
        self._startup_finished.clear()
        self._openai_tool_definitions.clear()
        self._shutdown_requested = asyncio.Event()

        self._background_thread = threading.Thread(
            target=self._run_background_loop,
            name="mcp-client",
            daemon=True,
        )
        self._background_thread.start()

        if not self._startup_finished.wait(self._timeout_seconds):
            self.close()
            raise TimeoutError(f"连接 MCP Server 超时: {self._server_url}")
        if self._startup_exception is not None:
            error = self._startup_exception
            self.close()
            raise error

        print(
            f"连接 MCP Server 成功: {self._server_url}, "
            f"获取到 {len(self._openai_tool_definitions)} 个工具"
        )
        return list(self._openai_tool_definitions)

    def _run_background_loop(self) -> None:
        """后台线程入口：asyncio.run 负责创建事件循环，并在退出时自动关闭。"""
        asyncio.run(self._maintain_session())

    async def _maintain_session(self) -> None:
        """后台协程：建立连接 → 发现工具 → 保持存活等待调用。"""
        # 记录当前循环，供主线程通过 run_coroutine_threadsafe 提交工具调用
        self._background_loop = asyncio.get_running_loop()
        try:
            # 建立 Streamable HTTP 连接。
            # read：读取 MCP Server 消息。
            # write：向 MCP Server 发送消息。
            # async with 退出时会自动清理 HTTP 连接。
            async with streamable_http_client(self._server_url) as (read, write):
                async with ClientSession(read, write) as session:   # 基于读写流创建 MCP 会话。
                    await session.initialize()      #  MCP 初始化握手，确认客户端和服务端的协议能力。
                    self._client_session = session
                    tool_result = await session.list_tools()    # 向 MCP Server 请求所有可用工具。
                    self._openai_tool_definitions.extend(
                        mcp_tools_to_openai(tool_result.tools)
                    )
                    self._startup_finished.set()
                    # 保持连接存活，直到 close() 请求关闭。
                    await self._shutdown_requested.wait()
        except Exception as exc:
            self._startup_exception = exc
            self._startup_finished.set()

    def call_tool(self, name: str, arguments: dict) -> str:
        """调用 MCP 工具，返回 JSON 字符串结果。"""
        if self._client_session is None or self._background_loop is None:
            return json.dumps({"error": "MCP 客户端未连接"}, ensure_ascii=False)

        # 当前线程是同步线程，但 session.call_tool() 是异步协程，所以不能直接调用。这里做了三件事：
        # 1. 创建工具调用协程。
        # 2. 把协程提交到后台事件循环。
        # 3. 返回一个跨线程 Future。
        call_future = asyncio.run_coroutine_threadsafe(
            self._client_session.call_tool(name, arguments),
            self._background_loop,
        )
        try:
            call_result = call_future.result(self._timeout_seconds)
        except Exception as exc:
            call_future.cancel()
            return json.dumps({"error": f"MCP 工具调用出错: {exc}"}, ensure_ascii=False)

        text = self._extract_text_content(call_result)
        if call_result.is_error:
            return json.dumps({"error": f"工具执行出错: {text or '未知错误'}"}, ensure_ascii=False)
        return text or "{}"

    @staticmethod
    def _extract_text_content(call_result) -> str:
        """收集 MCP 返回的文本内容（忽略图片等非文本块）。"""
        texts = [
            item.text
            for item in call_result.content or []
            if getattr(item, "type", None) == "text" and getattr(item, "text", None)
        ]
        return "\n".join(texts)

    def close(self) -> None:
        """关闭 MCP 连接，清理后台线程。"""
        if self._background_loop is not None and not self._background_loop.is_closed():
            # 传 bound method，不能在本线程直接 set()（否则会往事件循环塞 None 回调）
            # call_soon_threadsafe() 的作用是：把 set() 安全地安排到后台事件循环中执行。
            with suppress(RuntimeError):
                self._background_loop.call_soon_threadsafe(
                    self._shutdown_requested.set
                )
        if self._background_thread is not None:
            self._background_thread.join(self._timeout_seconds)
        self._client_session = None
        self._background_loop = None
        self._background_thread = None
