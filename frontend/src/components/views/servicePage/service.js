import React, {useEffect, useState} from 'react'
import axios from 'axios'
import { BiSolidSend } from "react-icons/bi";

import NavBar from '../navBar/navBar'
import Footer from '../footer/footer'
import InfoList from './infoList'
import './service.css'

const getReturnedParam = (hash) => {
    const strAfterHash = hash.substring(1)
    const paramsInUrl = strAfterHash.split("&")
    const paramSplit = paramsInUrl.reduce((acc,currentValue)=>{
        const [key, value] = currentValue.split("=")
        acc[key] = value
        return acc
    }, {})
    return paramSplit
}

function Service(props) {
    const [Chat, setChat] = useState("")
    const [ChatList, setChatList] = useState([])
    const [Playlist, setPlaylist] = useState([])
    const [Playlists, setPlaylists] = useState([])
    const [Login, setLogin] = useState(false)

    useEffect(()=> {
        //get token and send to backend
        if(window.location.hash) {
            const {access_token, expires_in, token_type} = getReturnedParam(window.location.hash)
            localStorage.setItem("accessToken", access_token)
            localStorage.setItem("expires_in", expires_in)
            localStorage.setItem("token_type", token_type)
            const token_info = {
                access_token: access_token,
                expires_in: expires_in,
                token_type: token_type
            }
            sendToken(token_info)
            setLogin(true)
        } else {
            console.log("guest login")
        }
    }, [])

    useEffect(()=> {
        setPlaylists(Playlists.concat([Playlist]))
    }, [Playlist])

    //post access tocken
    const sendToken = (token_info) => {
        axios.post('http://localhost:8000/login', token_info)
        .then(response => {
            if(response.data.success){
                console.log("succes to login")
            }else{
                alert('fail to login')
            }
        })
    }
    
    //챗 내용 저장
    const handleClick = (event) => {
        setChat(event.currentTarget.value)
    }

    //태그 리스트(나중에 추천 모델 연결)
    const tags = ["겨울", "인디", "잔잔한", "휴식", "팝", "이지리스닝"]

    //do tag based recommendation
    const clickTag = (event) => {
        return
    }

    //tag chat component
    const tag_list = tags.map((tag, index)=>{
        return (
            <div key={index}>
                <button className='tag' onClick={clickTag}>{tag}</button>
            </div>
        )
    })

    //추천 플레이리스트 가져오기
    const getPlaylist = (chat) => {
        console.log("get playlist")
        axios.put('http://localhost:8000/recommend', {"chat": chat})
        .then(response => {
            if(response.data.success){
                console.log("succes to get playlist")
                setPlaylist(response.data.playlist)
            }else{
                console.log('fail to get playlist')
            }
        }) 
    }

    const onSubmit = (event) => {
        event.preventDefault()   //prevent refreshing when click submit
        setChatList(ChatList.concat(Chat))
        localStorage.setItem("chat", Chat)
        localStorage.setItem("chatList", ChatList)
        getPlaylist(Chat)
        setChat("")
    }

    return (
        <div className='page'>
            <NavBar />
            <div className='chatbox'>
                <InfoList tags={tags} chats={ChatList} playlists={Playlists} login={Login}/>
            </div>
            <form onSubmit={onSubmit} className='chatform'>
                <textarea className='enterChat'
                    onChange={handleClick}
                    value={Chat}
                    rows={1}
                    style={{overflow:'hidden'}}
                />
                <BiSolidSend className='chatSubmit' onClick={onSubmit} size={30}/>
            </form>
            <div className='tagList'>
                {tag_list}
            </div>
            <Footer />
        </div>
    )
}

export default Service