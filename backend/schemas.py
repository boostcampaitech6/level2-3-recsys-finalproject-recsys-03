from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    # refresh_toekn:str
    expires_in:str
    token_type:str

class ChatRequest(BaseModel):
    chat:str
    user_uri:str
    type:str

class Track(BaseModel):
    title:str
    artist:str
    uri:str

class recommendResponse(BaseModel):
    success:bool
    music:list[Track]

class User(BaseModel):
    user_uri: str

class FeedbackRequest(BaseModel):
    playlist:list[str]
    user_uri: str