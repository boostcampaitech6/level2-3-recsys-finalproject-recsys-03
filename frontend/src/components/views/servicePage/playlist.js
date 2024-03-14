import React, { useEffect, useState } from 'react'
import { MdOutlineRemoveCircleOutline } from "react-icons/md";

import Play from './play'
import Export from './export'
import './infoList.css'

function Playlist(props) {
    const [Pause, setPause] = useState(true)
    const [Active, setActive] = useState(false)
    const [CurrentTrack, setTrack] = useState([])
    const [Token, setToken] = useState(localStorage.getItem("accessToken"))
    const [DeviceId, setDeviceId] = useState("")
    const [Playlist, setPlaylist] = useState(props.playlist)
    const login = props.login

    useEffect(() => {
        
        if (login) {
            //create web player
            const script = document.createElement("script");
            script.src = "https://sdk.scdn.co/spotify-player.js";
            script.async = true;
            document.body.appendChild(script);

            window.onSpotifyWebPlaybackSDKReady = () => {

                const player = new window.Spotify.Player({
                    name: "Web Playback SDK",
                    getOAuthToken: cb => { cb(Token) },
                    volume: 0.5,
                });

                player.addListener("ready", ({ device_id }) => {
                    console.log("Ready with Device ID", device_id)
                    setDeviceId(device_id)
                });

                player.addListener("not_ready", ({ device_id }) => {
                    console.log("Device ID has gone offline", device_id)
                });
                player.addListener("player_state_changed", (state) => {
                    if (!state) {
                        return
                    }

                    setTrack(state.track_window.current_track)
                    setPause(state.paused);

                    player.getCurrentState().then(state => {
                        (!state) ? setActive(false) : setActive(true)
                    })
                    console.log("state changed", state)
                })

                player.connect();
            }
        }
    }, [])


    const remove_song = (index) => {
        let new_playlist = [...Playlist.slice(0, index), ...Playlist.slice(index + 1)]
        setPlaylist(new_playlist)
    }

    //playlist component
    const playlist_song = Playlist.map((song, index)=>{
        return(
            <div key={index}>
                <div className='song'>
                    <div className='song_info'>
                        <h3 className='title'>{song.title}</h3>
                        <h3 className='artist'>{song.artist}</h3>
                    </div>
                    {login &&
                        <Play current_uri={song.uri} playlist={Playlist} device_id={DeviceId} />
                    }
                    <MdOutlineRemoveCircleOutline className='remove' onClick={() => remove_song(index)} size={25}/>
                </div>
            </div>
        )
    })

    return(
        <div className='playlist_box'>
            <div className='playlist_head'>
                <h2>New Playlist</h2>
                <Export playlist={Playlist} login={props.login}/>
            </div>
            <div className='playlist_content'>
                {playlist_song}
            </div>
        </div>
    )




}

export default Playlist