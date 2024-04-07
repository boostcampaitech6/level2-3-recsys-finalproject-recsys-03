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
                        <Playlist playlist={props.playlists[index+1]} login={props.login} user_uri={props.user_uri}/>
                    </div>
                }
                {(props.playlists.length > index+1 && !props.login) &&
                        <div className='service_chat'>
                            <h3 className='chat'>개인화 추천과 더 많은 서비스 이용을 위해 스포티파이 계정으로 로그인 해주세요!</h3>
                        </div>
                    }
            </div>
        )
    })

    
    return(
        <div className='chats'>
            <div className='service_chat'>
                <p className='sender'>Chat</p>
                <h3 className='chat'>안녕하세요, 당신이 원하는 플레이리스트는 무엇인가요?</h3>
            </div>
            {recommendation}
        </div>
    )

}

export default InfoList