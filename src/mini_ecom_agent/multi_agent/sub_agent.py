import json

from openai import OpenAI

from mini_ecom_agent.prompts.agents import COMPLAINT_PROMPT,POSTSALE_PROMPT, PRESALE_PROMPT
from mini_ecom_agent.tools.manager import ToolManager

AGENT_CONFIG = {
    "presale":{
        "name": "小夕-售前",
        "prompt":PRESALE_PROMPT,
        "tools": {"query_product", "search_knowledge", "list_user_orders", "load_skill"},
    },
    "postsale": {
        "name": "小夕-售后",
        "prompt": POSTSALE_PROMPT,
        "tools": {"query_order", "query_logistics", "apply_refund","list_user_orders", "search_knowledge", "load_skill"},
    },
    "complaint": {
        "name": "小夕-投诉",
        "prompt": COMPLAINT_PROMPT,
        "tools": {"query_order", "search_knowledge", "load_skill"},
    }
}


class SubAgent:
    def __init__(self, name, system_prompt, tool_manager:ToolManager,client:OpenAI, model:str, temperature:float):
        self.name = name
        self.system_prompt = system_prompt
        self.tools_manager = tool_manager
        self.client = client
        self.model = model
        self.temperature = temperature


    def handle(self, message:list[dict], max_steps: int):
        """
        ReAct循环
        """
        new_message = []
        working = list(message)

        for step in range(max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=working,
                tools=self.tools_manager.tool_definitions,
                temperature=self.temperature,
            )

            assistant_msg = response.choices[0].message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                return assistant_msg.content or "", new_message

            # 处理有工具调用情况
            msg_dict:dict = {"role":"assistant", "content":assistant_msg.content}
            msg_dict['tool_calls'] = [{
                "id":tc.id,
                "type":"function",
                "function":{
                    "name":tc.function.name,
                    "arguments":tc.function.arguments,
                }} for tc in assistant_msg.tool_calls
            ]

            new_message.append(msg_dict)
            working.append(msg_dict)

            for tc in assistant_msg.tool_calls:
                fun_name = tc.function.name
                fun_args = json.loads(tc.function.arguments)
                self._print_action(fun_name, fun_args)
                result_str = self.tools_manager.execute_tools(fun_name, fun_args)
                self._print_observation(result_str)

                tool_msg = {
                    "role":"tool",
                    "tool_call_id":tc.id,
                    "content":result_str,
                }
                new_message.append(tool_msg)
                working.append(tool_msg)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=working,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return content, new_message

    def _print_thought(self, text: str) -> None:
        print(f"\n  💭 [{self.name}·思考] {text}")

    def _print_action(self, func_name: str, func_args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
        print(f"  🔧 [{self.name}·调用工具] {func_name}({args_str})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        print(f"  📋 [{self.name}·工具结果] {display}")
