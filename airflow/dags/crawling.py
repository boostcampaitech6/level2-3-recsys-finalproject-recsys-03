import requests
import urllib.request as ur
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By

import boto3
import base64
import json
from tqdm import tqdm
import time

from pymongo import MongoClient
import threading

global secret
def get_new_interaction_track():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='prod/WebBeta/MongoDB')
    secret = json.loads(response['SecretString'])
    mongodb_uri = secret['mongodb_uri']
    
    client = MongoClient(mongodb_uri)
    db = client['playlist_recommendation']
    collection = db['User']
    
    data = collection.find({})
    new_track_uris = set()
    for user in data:
        for track_uri in user['new_track']:
            new_track_uris.add(track_uri)    
            
    return list(new_track_uris)

def fetch_audio_info(headers, params):
    audio_info = requests.get(f"https://api.spotify.com/v1/audio-features", headers=headers, params=params)
    audio_info = eval(audio_info.text)
    audio_info = audio_info["audio_features"]
    return audio_info

def fetch_basic_info(headers, params):
    basic_info = requests.get(f"https://api.spotify.com/v1/tracks", headers=headers, params=params)
    basic_info = eval(basic_info.text)
    tracks = basic_info["tracks"]
    basic_info = [{'track_name': track.get('name'), 'artist_name': track.get('artists')[0]['name']} for track in tracks]
    return basic_info

def get_audio_features(uris: list):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='prod/WebBeta/MongoDB')
    secret = json.loads(response['SecretString'])
    client_id = secret['client_id']
    client_secret = secret['client_secret']
    
    endpoint = "https://accounts.spotify.com/api/token"
    
    encoded = base64.b64encode("{}:{}".format(client_id, client_secret).encode('utf-8')).decode('ascii')

    headers = {"Authorization": "Basic {}".format(encoded)}
    payload = {"grant_type": "client_credentials"}
    response = requests.post(endpoint, data=payload, headers=headers)
    
    # api 호출하여 json 데이터 받기
    headers = {"Authorization": "Bearer {}".format(access_token)}
    
    metas = ["danceability","energy","key","loudness","mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]
    
    start = time.time()
    access_token = json.loads(response.text)['access_token']
    new_track_uris = get_new_interaction_track()
    
    data = []
    for uri in new_track_uris:
        new_item = {'uri': uri}
        data.append(new_item)
        
    ## Spotify Search API
    for i in tqdm(range(0,len(new_track_uris), 100)):
        end = time.time()
        music_uris = ','.join([music for music in new_track_uris])
        params = {
            "ids":music_uris
        }
        
        if end-start>=3600:
            print(i, "th saved and token expired")
            break
        if audio_info.status_code == 200 and basic_info.status_code == 200:
            
            audio_thread = threading.Thread(target=fetch_audio_info, args=(headers, params))
            basic_thread = threading.Thread(target=fetch_basic_info, args=(headers, params))
            audio_thread.start()
            basic_thread.start()
            
            audio_thread.join()
            basic_thread.join()
            audio_info = audio_thread.result
            basic_info = basic_thread.result
            
            data = [{**d1,**d2,**d3} for d1,d2,d3 in zip(data, audio_info, basic_info)]
        
            time.sleep(4)
        else:
            
            print(f"{response.status_code} error occurs")
            break

    return data
    
def get_tags_with_lastfm_api(tracks: list):
    api_key = secret['lastfm_api']
    
    for i in tqdm(len(tracks)):
        url=f"https://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={tracks[i]['artist_name']}&track={tracks[i]['track_name']}&api_key={api_key}&format=json"
        url = url.replace(" ","+")
        jsonurl = ur.urlopen(quote(url, safe=':/?&='))
        data = jsonurl.read()
        data = json.loads(data)
        
        try:
            origin_tags = data["toptags"]["tag"]
            result_tags = []
            for tag in origin_tags:
                result_tags.append(tag['name'])
            
            tracks[i]["tags"] = result_tags
        except:
            continue
    return tracks

def get_tags_with_crawling_lastfm(tracks: list):
    driver = webdriver.Chrome()
    for i in tqdm(range(len(tracks))):
        if len(tracks[i]['tags'])<=2:
            url = f'https://www.last.fm/search/tracks?q={tracks[i]["artist_name"]}+{tracks[i]["track_name"]}'
            url = url.replace(" ", "+")
            driver.get(url)
            time.sleep(2)
            driver.find_element(By.XPATH, '//*[@id="mantle_skin"]/div[3]/div/div[1]/table/tbody/tr[1]/td[4]/a').click()
            time.sleep(2)
            tags = driver.find_elements(By.CLASS_NAME,'tag')
            tags = set(tracks[i]['tags']+[tag.text for tag in tags])
            i=2
            while len(list(tags))<=6:
                driver.back()
                time.sleep(1)
                try:
                    driver.find_element(By.XPATH, f'//*[@id="mantle_skin"]/div[3]/div/div[1]/table/tbody/tr[{i}]/td[4]/a').click()
                    time.sleep(2)
                    new_tags = driver.find_elements(By.CLASS_NAME,'tag')
                    for tag in new_tags:
                        tags.add(tag.text)
                    i+=1
                except:
                    break
            tracks[i]['tags'] = list(tags)
    return tracks