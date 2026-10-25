"""记忆查询工具：让 Agent 在 ReAct 循环中主动查询用户记忆。"""
from mini_ecom_agent.agents.memory.manager import MemoryManager


_memory_manager : MemoryManager | None = None

def set_memory_manager(manager:MemoryManager):
    """由 Agent 初始化时调用，注入 MemoryManager 引用。"""
    global _memory_manager
    _memory_manager = manager

def recall_user_memory(query:str=''):
    """查询当前用户的记忆信息（长期记忆和短期记忆）。"""

    if _memory_manager is None or not _memory_manager.memory_enabled:
        return {"success": False, "error": "记忆系统未启用"}

    result = {
        'success': True,
        'short_term_memory':_memory_manager.stm.facts,
        'long_term_memory':[
            {'content':f.content,'category':f.category} for f in _memory_manager.ltm.facts
        ]
    }

    if _memory_manager.ltm.interaction_summaries:
        result['interaction_summaries'] = [
            s['summary'] for s in _memory_manager.ltm.interaction_summaries[-3:]
        ]

    return result
