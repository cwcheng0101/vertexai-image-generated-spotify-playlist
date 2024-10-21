from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import json

class PlaylistCurationInput(BaseModel):
    playlist_name: str = Field(..., description="Name of the playlist")
    playlist_description: str = Field(..., description="Description of the playlist")
    tracks: List[Dict[str, Any]] = Field(..., description="List of tracks with their features")
    summary: str = Field(..., description="Summary of how the curation was created")

class PlaylistCurationFormatterTool(BaseTool):
    name: str = "PlaylistCurationFormatterTool"
    description: str = "A tool that formats playlist curation data into a structured JSON format."
    args_schema: Type[BaseModel] = PlaylistCurationInput

    def _run(
        self,
        playlist_name: str,
        playlist_description: str,
        tracks: List[Dict[str, Any]],
        summary: str,
    ) -> str:
        formatted_curation = {
            "playlist_name": playlist_name,
            "playlist_description": playlist_description,
            "tracks": [
                {
                    "song_name": track.get("song_name", ""),
                    "artists": track.get("artists", ""),
                    "id": track.get("id", ""),
                    "features": {
                        key: value for key, value in track.items() 
                        if key not in ["song_name", "artists", "id"]
                    }
                }
                for track in tracks
            ],
            "summary": summary
        }
        return json.dumps(formatted_curation, indent=2)

    async def _arun(self, playlist_name: str, playlist_description: str, tracks: List[Dict[str, Any]], summary: str) -> str:
            return self._run(playlist_name, playlist_description, tracks, summary)