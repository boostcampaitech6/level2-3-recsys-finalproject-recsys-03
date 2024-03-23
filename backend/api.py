from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from config import config
from authlib.integrations.starlette_client import OAuth, OAuthError
from pymongo import MongoClient 
from schemas import Token, ChatRequest, Track, User
from make_playlist import make_playlist
import pandas as pd
from httpx import AsyncClient
from datetime import datetime
import requests
import time


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
    user = await get_spotify(url = "https://api.spotify.com/v1/me", token=token_info.access_token)
    top_items = await get_spotify(url = "https://api.spotify.com/v1/me/top/tracks?time_range=long_term&limit=50", token=token_info.access_token)

    # MongoDB 연결
    client = MongoClient(config.db_url)
    db = client['playlist_recommendation']
    users_collection = db['User']
    tracks_collection = db['Track']
    listening_collection = db['Listening Events']

    existing_user = users_collection.find_one({'uri':user['id']})

    if existing_user:
        for item in top_items['items']:
            track = tracks_collection.find_one({'uri':item['id']})
            track_id = track['track_id'] if track else -1
            
            music = {
                'uri':item['id'],
                'track_id':track_id
            }

            if music not in existing_user.get('top_track', []):
                users_collection.update_one({'uri': user['id']}, {'$push': {'top_track': music}})
            if track_id!=-1:
                listening_collection.update_one(
                    {'user_id' : existing_user['user_id']},
                    {'$addToSet': {'track_id': track_id}},
                    upsert=True
                )
    else:
        top_item_list = []
        listening_list = []
        for item in top_items['items']:
            # artist 여러명인 경우 처리
            # artists = [artist['name'] for artist in item['artists']]
            
            # track_id 조회
            track = tracks_collection.find_one({'uri':item['id']})
            track_id = track['track_id'] if track else -1
            if track_id!=-1:
                listening_list.append(track_id)

            music = {
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

        if listening_list:
            listening_event = {
                'user_id':last_user['user_id'] + 1,
                'track_id':listening_list
            }
            listening_collection.insert_one(listening_event)
    return JSONResponse(content={"success": True, "message": "Operation successful", "uri" : user["id"]})

@router.put('/tags')
async def recommend_displayed_tags(user: User):
    user_uri = user.user_uri
    # 비회원인 경우
    if user_uri == "":
        tag_list = ["winter", "pop", "energetic", "sadness", "00s", "singer songwriter", "piano"]
    else:
        # MongoDB 연결
        client = MongoClient(config.db_url)
        db = client['playlist_recommendation']
        users_collection = db['User']
        tracks_collection = db['Track']
        
        user_collection = users_collection.find_one({'uri':user_uri})
        top_items = user_collection["top_track"]
        existed_top_items = []
        for item in top_items:
            if item['track_id'] != -1:
                existed_top_items.append(item['track_id'])
                
        if len(existed_top_items)>=10:
            tag_counter = dict()
            for item in existed_top_items:
                track = tracks_collection.find_one({'track_id':item})
                for tag in track["tags"]:
                    if tag in tag_counter.keys():
                        tag_counter[tag] += 1
                    else:
                        tag_counter[tag] = 1
            tag_sorted = dict(sorted(tag_counter.items(), key=lambda item: item[1], reverse=True))
            tag_list = list(tag_sorted.keys())[:7]
        else:
            tag_list = ["winter", "pop", "energetic", "sadness", "00s", "singer songwriter", "piano"]
        
    # 현재 월, 시간, 날씨를 기반으로 태그 추천
    now = datetime.now()
    if now.month>=3 and now.month<=5:
        tag_list += ['spring']
    elif now.month>=6 and now.month<=8:
        tag_list += ['summer']
    elif now.month>=9 and now.month<=10:
        tag_list += ['autumn']
    else:
        tag_list += ['winter']
        
    if now.hour>=6 and now.hour<=9:
        if now.hour<=7:
            tag_list += ['wake up']
        tag_list += ['morning']
    elif now.hour in [22, 23, 0, 1]:
        tag_list += ['night']
    
    api_key = config.weathermap_api_key
    city = "Seoul"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data["cod"] == 200:
        if data['weather'][0]['main'] == 'Rain':
            tag_list += ['rainy day']
        elif data['weather'][0]['main'] == 'Snow':
            tag_list += ['snow']
    return JSONResponse(content={"success": True, "tag_list": tag_list})
    
@router.put('/recommend')
async def recommend_tag(chatRequest:ChatRequest):
    start = time.time()
    chat = chatRequest.chat
    user_uri = chatRequest.user_uri

    client = MongoClient(config.db_url)
    db = client['playlist_recommendation']
    user_chat_db = db['User_Chat']
    
    user_chat = {
        'user': user_uri,
        'chat' : chat
    }
    user_chat_db.insert_one(user_chat)

    df_tags = pd.read_csv('../data/tag_list.csv')
    # 최종적으로 올린 23000개 tag_list로 일단 작업해두겠습니당 (SBK)
    tags = df_tags.tag

    # titles, artists, uris = make_playlist(chat, tags)
    # for title, artist, uri in zip(titles, artists, uris):
    #     track = Track(title=title, artist=artist, uri=uri).model_dump()
    #     playlist.append(track)
    # if not titles:
    #     return JSONResponse(content={"success": False, "message": "Can't get recommend result"})
    playlist = make_playlist(chat, user_uri, tags)
    for item in playlist:
        item['uri'] = "spotify:track:" + item['uri']
        
    # print(playlist)
    end = time.time()
    print(f"{end - start:.5f} sec")
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


