from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]