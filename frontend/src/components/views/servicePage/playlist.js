import React, { useEffect, useState } from 'react'
import { MdOutlineRemoveCircleOutline } from "react-icons/md";
import { MdOutlinePlayCircle } from "react-icons/md"

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
    const [ImageList, setImageList] = useState([])
    const login = props.login

    useEffect(()=>{
        if (login) {
            //get album art
            let ids = props.playlist.map(song => song.uri.split(":")[2])
            get_image(Token, ids)
        }
    }, [])

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

    const get_image = (access_token, ids) => {
        fetch(`https://api.spotify.com/v1/tracks?ids=${ids}`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${access_token}`,
                'Content-Type': 'application/json'
            },
        }).then(response => {
            if(response.ok){
                console.log("get track info")
                return response.json()
            }else{
                console.log("fail to get track info")
            }
        }).then(data => {
            const uris = data.tracks.map(track => track.album.images[0].url);
            setImageList(uris)
        })
    }


    const remove_song = (index) => {
        let new_playlist = [...Playlist.slice(0, index), ...Playlist.slice(index + 1)]
        setPlaylist(new_playlist)
    }

    //playlist component
    const playlist_song = Playlist.map((song, index)=>{
        let link = 'https://www.youtube.com/results?search_query='
        let search = (song.artist + '+' + song.title).replace(' ', '+')
        return(
            <div key={index}>
                <div className='song'>
                    {login && 
                        <div className='album_art'>
                            <img src={ImageList[index]}/>
                        </div>
                    }
                    <div className='song_info'>
                        <h3 className='title'>{song.title}</h3>
                        <h3 className='artist'>{song.artist}</h3>
                    </div>
                    {login &&
                        <Play song_uri={song.uri} playlist={Playlist} device_id={DeviceId} current_track={CurrentTrack}/>
                    }
                    {!login &&
                        <a href={link+search} target="_blank" rel="noopener noreferrer"><MdOutlinePlayCircle className='play' size={25}/></a>
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
            </div>
            <div className='playlist_content'>
                {playlist_song}
            </div>
            <Export playlist={Playlist} login={props.login} user_uri={props.user_uri}/>
        </div>
    )




}

export default Playlist