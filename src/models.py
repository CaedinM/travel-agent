from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from dataclasses import dataclass

@dataclass
class Resources:
    flights_agent: object
    flights_tools_by_name: dict[str, BaseTool]


class BestOffer(BaseModel):
    """The subagent's answer: which offer it chose."""
    offer_id: str = Field(description="The exact offer_id string copied from a search-flights result. Never invent or reformat one.")
    reasoning: str = Field(description="One sentence on why this offer beats the alternatives.")