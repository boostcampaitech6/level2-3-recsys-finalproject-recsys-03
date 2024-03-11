from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    # refresh_toekn:str
    expires_in:str
    token_type:str

class ChatRequest(BaseModel):
    chat:str

class Track(BaseModel):
    title:str
    artist:str
    uri:str

class recommendResponse(BaseModel):
    success:bool
    music:list[Track]
