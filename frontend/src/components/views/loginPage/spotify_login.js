import React, {useEffect, useState} from 'react'
import axios from 'axios'
import {useNavigate} from 'react-router-dom'
import {SPOTIFY_AUTHORIZE_ENDPOINT, CLIENT_ID, REDIRECT_URL, SCOPES_PARAM} from './config'
import './login.css'



 

function Spotify_login(props) {
    let navigate = useNavigate()
    
    const handleLogin = () => {
        window.location = `${SPOTIFY_AUTHORIZE_ENDPOINT}?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URL}&scope=${SCOPES_PARAM}&response_type=token&show_dialog=true`
    }

    const guestLogin = () => {
        navigate('/service')
    }

    return (
        <div className='login'>
            <h1>Suggestify</h1>
            <div className='buttons'>
                <button onClick={handleLogin}>Login with Spotify</button>
                <button onClick={guestLogin}>Guest Login</button>
            </div>
        </div>
    )
}

export default Spotify_login