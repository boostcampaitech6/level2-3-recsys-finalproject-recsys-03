from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from config import config
from authlib.integrations.starlette_client import OAuth, OAuthError
from pymongo import MongoClient 
from schemas import Token, ChatRequest, Track
from make_playlist import make_playlist
import pandas as pd
from httpx import AsyncClient

router = APIRouter()

CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
REDIRECT_URI = config.redirect_uri
AUTHENTICATION_URI = "https://accounts.spotify.com"

oauth = OAuth()
oauth.register(
    name='spotify',
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    access_token_url='https://accounts.spotify.com/api/token',
    authorize_url='https://accounts.spotify.com/authorize',
    api_base_url='https://api.spotify.com/v1/',
    client_kwargs={
        'scope': 'user-read-private user-read-email user-read-recently-played user-top-read',
    },
)

async def get_spotify(url, token):
    auth_headers = {
        "Authorization": f"Bearer {token}"
    }
    async with AsyncClient() as client:
        response = await client.get(url, headers=auth_headers)
        if response.status_code != 200:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Could not fetch Spotify API")
    return response.json()

@router.post('/login', status_code=201)
async def login(token_info:Token):
    print(token_info.access_token)
    user = await get_spotify(url = "https://api.spotify.com/v1/me", token=token_info.access_token)
    top_items = await get_spotify(url = "https://api.spotify.com/v1/me/top/tracks?time_range=long_term&limit=50", token=token_info.access_token)

    # MongoDB 연결
    client = MongoClient(config.db_url)
    db = client['playlist_recommendation']
    users_collection = db['User']
    tracks_collection = db['Track']
    listening_collection = db['Listening Events']

    if not users_collection.find_one({'uri':user['id']}):
        top_item_list = []
        listening_list = []
        for item in top_items['items']:
            # artist 여러명인 경우 처리
            artists = item['artists']
            artist_list= []
            for artist in artists:
                artist_list.append(artist['name'])
            
            # track_id 조회
            track = tracks_collection.find_one({'uri':item['id']})
            track_id = -1
            if track:
                track_id = track['track_id']
                listening_list.append(track_id)

            music = {
                'artist_name':', '.join(artist_list),
                'track_name':item['name'],
                'uri':item['id'],
                'track_id':track_id
            }
            top_item_list.append(music)
        
        last_user = listening_collection.find_one(sort=[("user_id", -1)])

        user_data= {
            'uri': user['id'],
            'email': user['email'],
            'country': user['country'],
            'top_track':top_item_list,
            'user_id':last_user['user_id'] + 1
        }
        users_collection.insert_one(user_data)

        if len(listening_list)>0:
            listening_event = {
                'user_id':last_user['user_id'] + 1,
                'track_id':listening_list
            }
            listening_collection.insert_one(listening_event)
    return JSONResponse(content={"success": True, "message": "Operation successful"})

@router.put('/recommend')
async def recommend_tag(chatRequest:ChatRequest):
    chat = chatRequest.chat

    df_tags = pd.read_csv('../data/tag.csv')
    tags = df_tags.Tag
    playlist = []
    # titles, artists, uris = make_playlist(chat, tags)
    # for title, artist, uri in zip(titles, artists, uris):
    #     track = Track(title=title, artist=artist, uri=uri).model_dump()
    #     playlist.append(track)
    # if not titles:
    #     return JSONResponse(content={"success": False, "message": "Can't get recommend result"})
    playlist = make_playlist(chat, tags)
    for item in playlist:
        item['uri'] = "spotify:track:" + item['uri']
        
    print(playlist)
    
    return JSONResponse(content={"success": True, "playlist": playlist})

# TODO test
# @router.get('/refresh')
# async def refresh_token(request: Request) -> Token:
#     refresh_token = request.query_params.get('refresh_token')
#     if not refresh_token:
#         raise HTTPException(status_code=400, detail="Invalid authorization")
#     try:
#         new_token = await oauth.spotify.refresh_token(
#             url=oauth.spotify.access_token_url,
#             refresh_token=refresh_token,
#             grant_type='refresh_token'
#         )
#     except OAuthError as error:
#         raise HTTPException(status_code=400, detail=str(error))

#     if not new_token:
#         raise HTTPException(status_code=400, detail="Could not refresh token")

#     refresh_token = new_token.get('refresh_token', refresh_token)

#     # 새로운 액세스 토큰을 반환
#     return Token(access_token=new_token['access_token'], refresh_toekn=refresh_token)


