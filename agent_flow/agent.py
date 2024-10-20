from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

import os
from agent_flow.tools.spotify_playlist_tool import SpotifyPlaylistTool
from agent_flow.tools.spotify_features_tool import SpotifyFeaturesTool
from agent_flow.tools.image_preprocess_tool import ImagePreprocessTool

from agent_flow.assistant import Assistant
from agent_flow.state import AgentState

from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
load_dotenv()

class Agent:
    def __init__(self, spotify_token: str, user_id: str):
        self.spotify_token = spotify_token
        self.user_id = user_id
        self.model = self._initialize_model()
        self.tools = self._initialize_tools()
        self.graph = self._create_graph()
        self.image_preprocess_tool = ImagePreprocessTool()

    def _initialize_model(self):
        return ChatVertexAI(
            model="gemini-1.5-flash-001",
            temperature=0,
            max_tokens=None,
            max_retries=6,
            stop=None,
            project=os.getenv("PROJECT_ID"),
        )

    def _initialize_tools(self):
        return [
            SpotifyFeaturesTool(spotify_token=self.spotify_token),
            SpotifyPlaylistTool(user_id=self.user_id, spotify_token=self.spotify_token),
        ]

    @staticmethod
    def _handle_tool_error(state):
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    @classmethod
    def _create_tool_node_with_fallback(cls, tools):
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(cls._handle_tool_error)],
            exception_key="error"
        )

    def _create_graph(self):
        primary_assistant_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                '''You are an AI Agent that operates the Spotify API and image analysis tools to respond to User's requests.
                Execution:
                1. Use the SpotifyFeaturesTool to get audio features of recent tracks in user's account.
                2. According to the audio features data of recent tracks, use the SpotifyPlaylistTool to create a playlist based on the image analysis or text input
                3. Provide a summary of the created playlist and its relation to the input (image or text).
                '''
            ),
            ("placeholder", "{messages}"),
        ])

        part_1_assistant_runnable = primary_assistant_prompt | self.model.bind_tools(self.tools)

        builder = StateGraph(AgentState)
        builder.add_node("assistant", Assistant(part_1_assistant_runnable))
        builder.add_node("tools", self._create_tool_node_with_fallback(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def _preprocess_image(self, image_string: str) -> str:
        return self.image_preprocess_tool._run(image_string=image_string)

    def process_request(self, question: str, image_string: str = None):
        import uuid

        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # If an image is provided, preprocess it first
        if image_string:
            image_description = self._preprocess_image(image_string)
            question = f"Based on this image description: {image_description}\n\n{question}"

        events = self.graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        
        results = []
        for event in events:
            results.append(event)
        
        return results