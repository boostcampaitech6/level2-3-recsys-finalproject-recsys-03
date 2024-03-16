import React, { useEffect, useState } from 'react'
import { MdOutlinePlayCircle } from "react-icons/md"
import { MdOutlinePauseCircle } from "react-icons/md"

import './infoList.css'

const play = (uris, song_uri, id, access_token) => {
    fetch(`https://api.spotify.com/v1/me/player/play?device_id=${id}`, {
        method: 'PUT',
        headers: {
            'Authorization': `Bearer ${access_token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'uris': uris,
            'offset': {'uri': song_uri},
            'position_ms': 0
        })
    }).then(response => {
        if(response.ok){
            console.log("now playing...")
        }else{
            console.log('fail to play the music')
        }
    })
}

const pause = (id, access_token) => {
    fetch(`https://api.spotify.com/v1/me/player/pause?device_id=${id}`, {
        method: 'PUT',
        headers: {
            'Authorization': `Bearer ${access_token}`,
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if(response.ok){
            console.log("pause")
        }else{
            console.log('fail to pause')
        }
    })
}

function Play(props) {
    const [Pause, setPause] = useState(false)
    let playlist = props.playlist
    let song_uri = props.song_uri
    let token = localStorage.getItem("accessToken")
    let device_id = props.device_id
    let current_track = props.current_track
    let uris = playlist.map(song => song.uri)


    const click_play = () => {
        play(uris, song_uri, device_id, token)
        setPause(false)
    }

    const click_pause = () => {
        pause(device_id, token)
        setPause(true)
    }

    if (current_track.uri==song_uri && !Pause) {
        return(
            <MdOutlinePauseCircle className='play' onClick={click_pause} size={25}/>
        )
    } else {
        return(
            <MdOutlinePlayCircle className='play' onClick={click_play} size={25}/>
        )
    }
}

export default Play