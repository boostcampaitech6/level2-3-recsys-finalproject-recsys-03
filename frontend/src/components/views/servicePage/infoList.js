import React, {useEffect, useState} from 'react'
import './infoList.css'
import Playlist from './playlist'


function InfoList(props) {
    let chats = props.chats
    let playlists = props.playlists

    //user chat component
    const chat_list = chats.map((chat, index)=>{
        return(
            <div key={index}>
                <h3 className='chat'>{chat}</h3>
            </div>
        )
    })

    const recommendation = chats.map((section, index)=>{
        console.log(props.playlists)
        return(
            <div key={index}>
                <div>
                    <h3 className='chat'>{chats[index]}</h3>
                </div>
                {props.playlists.length > index+1 &&
                    <div>
                        <Playlist playlist={props.playlists[index+1]} login={props.login}/>
                    </div>
                }
            </div>
        )
    })

    
    return(
        <div>
            {recommendation}
        </div>
    )

}

export default InfoList