import React, {useEffect, useState} from 'react'
import axios from 'axios'
import { BiSolidSend } from "react-icons/bi";

import NavBar from '../navBar/navBar'
import Footer from '../footer/footer'
import InfoList from './infoList'
import './service.css'
import {useNavigate} from 'react-router-dom'

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
    const [UserUri, setUserUri] = useState("")
    const [Tags, setTags] = useState([])
    let navigate = useNavigate()


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
            getTags(UserUri)
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
                setUserUri(response.data.uri)
                getTags(response.data.uri)
            }else{
                alert('Fail to login')
                navigate('/')
            }
        })
    }
    

    //get tag list
    const getTags = (user_uri) => {
        axios.put('http://localhost:8000/tags', {"user_uri": user_uri})
        //axios.put('http://localhost:3000/api/tags', {"user_uri": user_uri})
        .then(response => {
            if(response.data.success){
                console.log("succes to get tags")
                console.log(response.data.tag_list)
                setTags(response.data.tag_list)
            }else{
                console.log('fail to get tags')
            }
        })
    }

    //do tag based recommendation
    const clickTag = (tag) => {
        setChat(tag)
        setChatList(ChatList.concat(tag))
        localStorage.setItem("chat", tag)
        localStorage.setItem("chatList", ChatList)
        getPlaylist(tag, UserUri)
        setChat("")
    }

    //tag recommendation
    const tag_list = Tags.map((tag, index)=>{
        return (
            <div key={index}>
                <button className='tag' onClick={() => clickTag(tag)}>{tag}</button>
            </div>
        )
    })

    //추천 플레이리스트 가져오기
    const getPlaylist = (chat, user_uri) => {
        console.log(chat)
        if (chat == "" && chat == " "){
            alert("Invalid input")
            navigate('/')
            return
        }
        console.log("get playlist")
        axios.put('http://localhost:8000/recommend', {"chat": chat, "user_uri": user_uri})
        .then(response => {
            if(response.data.success){
                console.log("succes to get playlist")
                setPlaylist(response.data.playlist)
            }else{
                console.log('fail to get playlist')
            }
        }) 
    }

    //챗 내용 저장
    const handleClick = (event) => {
        setChat(event.currentTarget.value)
    }

    const onSubmit = (event) => {
        event.preventDefault()   //prevent refreshing when click submit
        setChatList(ChatList.concat(Chat))
        localStorage.setItem("chat", Chat)
        localStorage.setItem("chatList", ChatList)
        getPlaylist(Chat, UserUri)
        setChat("")
    }

    return (
        <div className='page'>
            <NavBar />
            <div className='chatbox'>
                <InfoList tags={Tags} chats={ChatList} playlists={Playlists} login={Login}/>
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