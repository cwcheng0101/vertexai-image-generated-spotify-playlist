from typing import List
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
from agent_flow.tools.playlist_curation_formatter_tool import PlaylistCurationFormatterTool

from agent_flow.assistant import Assistant
from agent_flow.state import AgentState

import uuid
import json
import re

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
            PlaylistCurationFormatterTool(),
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
                Please follow the execution item below:
                STEP1: Use the SpotifyFeaturesTool to get audio features of recent tracks in user's account.
                STEP2: According to the audio features data of recent tracks, create a playlist curation based on the image analysis or text input.
                STEP3: Use the PlaylistCurationFormatterTool to format the curated playlist and summary into a structured JSON format.
                STEP4: Please return the json string output from PlaylistCurationFormatterTool at the final message
                
                Context Knowledge: Meaning and symbolism of the parameters returned by SpotifyFeaturesTool:
                {{
                    "acousticness": Confidence measure of whether the track is acoustic, example_value: 0.00242, value range: "0 - 1";
                    "danceability": How suitable a track is for dancing, example_value: 0.585;
                    "duration_ms":  Track duration in milliseconds, example_value: 237040;
                    "energy": Perceptual measure of intensity and activity, example_value: 0.842;
                    "id": Spotify ID for the track, example_value: 2takcwOaAZWiXQijPHIx7B;
                    "instrumentalness": Predicts if a track contains no vocals, example_value: 0.00686;
                    "key": The key the track is in, example_value: 9, value range: "-1 - 11";
                    "liveness": Presence of an audience in the recording, example_value: 0.0866;
                    "loudness": Overall loudness of a track in decibels (dB), example_value: -5.883;
                    "mode": Modality (major or minor) of a track, example_value: 0;
                    "speechiness": Presence of spoken words in a track, example_value: 0.0556;
                    "tempo": Estimated tempo of a track in BPM, example_value: 118.211;
                    "time_signature:: Estimated time signature, example_value: 4, value range: "3 - 7";
                    "type": Object type, allowed_values: audio_features;
                    "valence": Musical positiveness conveyed by a track, example_value: 0.428, value range: "0 - 1";
                }}

                }}
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
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # If an image is provided, preprocess it first
        if image_string:
            image_description = self._preprocess_image(image_string)
            question = f"Based on this image description: {image_description}\n\n What kinds of audio features and track name will be the perfect fit for the playlist resonating with image?\n\n{question}"

        '''
        events = self.graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
        )
        results = []
        for event in events:
            results.append(event)
            print(event)
        '''
        
        
        result = self.graph.invoke({"messages": ("user", question)}, config)
        final_result = result['messages'][-1].content
        #final_result = []
        #print("raw", final_result)
        # Remove Markdown code block delimiters if present
        final_result = re.sub(r'```json\s*', '', final_result)
        final_result = re.sub(r'\s*```', '', final_result)
        final_result = final_result.strip()

        print("Cleaned final result:", final_result)  # For debugging

        try:
            curation = json.loads(final_result)
            return curation
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")  # For debugging
            return {"error": f"Failed to parse curation JSON: {str(e)}", "raw_content": final_result}
