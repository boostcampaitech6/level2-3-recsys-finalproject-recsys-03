import React, {useEffect, useState} from 'react'
import axios from 'axios'
import {useNavigate} from 'react-router-dom'
import {SPOTIFY_AUTHORIZE_ENDPOINT, CLIENT_ID, REDIRECT_URL, SCOPES_PARAM} from './config'
import './login.css'



 

function Spotify_login(props) {
    let navigate = useNavigate()
    
    const handleLogin = () => {
        try{
            window.location = `${SPOTIFY_AUTHORIZE_ENDPOINT}?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URL}&scope=${SCOPES_PARAM}&response_type=token&show_dialog=true`
        } catch(err){
            alert("현재 스포티파이 로그인은 수동 등록이 필요합니다. 이용을 위해 운영진에게 연락 주세요.")
            navigate('/')
        }
        
    }

    const guestLogin = () => {
        navigate('/service')
    }

    return (
        <div className='login'>
            <h1>Au-Dionysos</h1>
            <h3>Don't ruin my mood!</h3>
            <p>Au-Dionysos는 당신의 상황, 감정, 취향을 반영한 플레이리스트를 생성해드립니다. <br/> 
                당신의 이야기를 들려주세요. 저희는 당신에게 공감해 멋진 노래를 선물해드릴게요.</p>
            <div className='buttons'>
                <button onClick={handleLogin}>Login with Spotify</button>
                <button onClick={guestLogin}>Guest Login</button>
            </div>
        </div>
    )
}

export default Spotify_login