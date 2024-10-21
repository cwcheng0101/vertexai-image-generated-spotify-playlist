from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun
)
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
import os
from dotenv import load_dotenv
load_dotenv()

class ImagePreprocessInput(BaseModel):
    image_string: str = Field(description="string of the image to be analyzed")

class ImagePreprocessTool(BaseTool):
    name: str = "ImagePreprocessTool"
    description: str = (
        '''
        A tool that analyzes an image and returns a description which address below asks:
        Ask1: What kind of music vibe, mood and musical themes could symbolize or expressed by this image?
        Ask2: What kinds of tone of the playlist will this image leads to?
        
        ***This tool requires the string data of an image file as input***
        '''
    )
    args_schema: Type[BaseModel] = ImagePreprocessInput
    model: ChatVertexAI = ChatVertexAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        max_retries=6,
        stop=None,
        project= os.getenv("PROJECT_ID"),
    )

    def _run(
        self,
        image_string: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this image and describe its content, mood, and any musical themes it might inspire:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_string}"},
                },
            ]
        )
        analysis = self.model.invoke([image_message])
        return analysis.content

    async def _arun(
        self,
        image_string: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the ImagePreprocessTool asynchronously."""
        return self._run(image_string)