
from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    sentiment: str
    keywords: List[str]
