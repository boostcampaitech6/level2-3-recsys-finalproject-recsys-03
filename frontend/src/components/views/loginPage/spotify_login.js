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
            <h1>Au-Dionysos</h1>
            <h3>Don't ruin my mood!</h3>
            <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit, 
                sed do eiusmod tempor incididunt ut labore et dolore magna 
                aliqua. Ut enim ad minim veniam, quis nostrud exercitation 
                ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
            <div className='buttons'>
                <button onClick={handleLogin}>Login with Spotify</button>
                <button onClick={guestLogin}>Guest Login</button>
            </div>
        </div>
    )
}

export default Spotify_login