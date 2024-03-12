import React, { useEffect, useState } from 'react'

const play = (uris, current_uri, id, access_token) => {
    fetch(`https://api.spotify.com/v1/me/player/play?device_id=${id}`, {
        method: 'PUT',
        headers: {
            'Authorization': `Bearer ${access_token}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            'uris': uris,
            'offset': {'uri': current_uri},
            'position_ms': 0
        })
    }).then(response => {
        if(response.success){
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
        if(response.success){
            console.log("pause")
        }else{
            console.log('fail to pause')
        }
    })
}

function Play(props) {
    const [Pause, setPause] = useState(true)
    let playlist = props.playlist
    let current_uri = props.current_uri
    let token = localStorage.getItem("accessToken")
    let device_id = props.device_id
    let uris = playlist.map(song => song.uri)


    const click_play = () => {
        play(uris, current_uri, device_id, token)
        setPause(false)
    }

    const click_pause = () => {
        pause(device_id, token)
        setPause(true)
    }

    if (Pause) {
        return(
            <div>
                <button className='play' onClick={click_play}>Play</button>
            </div>
        )
    } else {
        return(
            <div>
                <button className='play' onClick={click_pause}>Pause</button>
            </div>
        )
    }
}

export default Play