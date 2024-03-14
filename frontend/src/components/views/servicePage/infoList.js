import React, {useEffect, useState} from 'react'
import './infoList.css'
import Playlist from './playlist'


function InfoList(props) {
    let chats = props.chats
    let playlists = props.playlists

    const recommendation = chats.map((section, index)=>{
        return(
            <div key={index}>
                <div className='user_chat'>
                    <p className='sender'>You</p>
                    <h3 className='chat'>{chats[index]}</h3>
                </div>
                {props.playlists.length <= index+1 &&
                    <div className='service_chat'>
                        <p className='sender'>Chat</p>
                        <h3 className='chat'>Loading...</h3>
                    </div>
                }
                {props.playlists.length > index+1 &&
                    <div className='service_chat'>
                        <p className='sender'>Chat</p>
                        <Playlist playlist={props.playlists[index+1]} login={props.login}/>
                    </div>
                }
            </div>
        )
    })

    
    return(
        <div className='chats'>
            {recommendation}
        </div>
    )

}

export default InfoList