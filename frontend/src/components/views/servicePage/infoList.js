import React, {useEffect, useState} from 'react'
import './infoList.css'
import Playlist from './playlist'


function InfoList(props) {
    let tags = props.tags
    let chats = props.chats
    let playlist = props.playlist

    //do tag based recommendation
    const onSubmit = (event) => {
        return
    }

    //tag chat component
    const tag_list = tags.map((tag, index)=>{
        return (
            <div key={index}>
                <button className='tag' onClick={onSubmit}>{tag}</button>
            </div>
        )
    })

    //user chat component
    const chat_list = chats.map((chat, index)=>{
        return(
            <div key={index}>
                <h3 className='chat'>{chat}</h3>
            </div>
        )
    })

    
    return(
        <div>
            <div className='service_chat'>
                {tag_list}
            </div>
            {chats.length >= 1 &&
                <div className='user_chat'>
                    {chat_list}
                </div>
            }
            {playlist.length >= 1 &&
                <Playlist playlist={playlist} login={props.login}/>
            }

        </div>
    )

}

export default InfoList