from typing import Optional
from langchain.agents.agent import AgentExecutor
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
            
class AgentAsTool(BaseTool):
    """
    Enables an agent to be used as a tool. The name and description properties
    should be set on creation based on what the agent does.
    """
    agent: AgentExecutor
    """Must be specified, the agent behind the tool"""
    output_key: str = "output"
    """The name of the output key from the agent response to return as the tool result"""

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        result = self.agent(query)
        if self.output_key not in result:
            raise RuntimeError(f"Output key: {self.output_key} not in agents response")
        return str(result[self.output_key])
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")