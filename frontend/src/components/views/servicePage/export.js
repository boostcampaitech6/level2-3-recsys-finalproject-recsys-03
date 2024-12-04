// eslint-disable-next-line no-unused-vars
import React, {useEffect, useState} from 'react'
// eslint-disable-next-line no-unused-vars
import axios from 'axios'

import './infoList.css'
import spotify_logo from "../../images/white_icon.png"

const add_items = (playlist_id, access_token, playlist) => {
    fetch(`https://api.spotify.com/v1/playlists/${playlist_id}/tracks`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${access_token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "uris": playlist,
            "position": 0
        })
    }).then(response => {
        if(response.ok){
            console.log("success to add items")
            //alert("saved the new playlist!")
            alert("새로운 플레이리스트가 추가되었습니다!")
        }else{
            console.log("fail to add items")
        }
    })
}


const create_playlist = (user_id, access_token, playlist) => {
    fetch(`https://api.spotify.com/v1/users/${user_id}/playlists`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${access_token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "name": "New Playlist",
            "description": "New playlist description",
        })
    }).then(response => {
        if(response.ok){
            console.log("success to create playlist")
            return response.json()
        }else{
            console.log("fail to create playlist")
        }
    }).then(data => {
        add_items(data.id, access_token, playlist)
    })
}

function Export(props) {
    const [Token, setToken] = useState(localStorage.getItem("accessToken"))
    const login = props.login
    const user_uri = props.user_uri
    let playlist = props.playlist.map(song => song.uri)
    
    //implicit feedback 받기
    const exported_items = (playlist, user_id) => {
        console.log(playlist, user_id)
        axios.post('https://au-dionysos.com/api/feedback', {"playlist": playlist, "user_uri": user_id})
        .then(response => {
            if(response.data.success){
                console.log("succes to save feedback")
            }else{
                console.log('fail to save feedback')
            }
        })
    }


    const handleClick = (event) => {
        //get user id -> create playlist -> add items
        create_playlist(user_uri, Token, playlist)
        exported_items(playlist, user_uri)
    }

    return(
        <div className='save_container'>
            {login &&
                <div className='save' onClick={handleClick}>
                    <h3>PLAY ON SPOTIFY</h3>
                    <img src={spotify_logo} alt=""></img>
                </div>
            }
        </div>
    )
}

export default Export