/* eslint-disable jsx-a11y/anchor-is-valid */
import React from 'react';
import axios from 'axios';
import { useSelector } from "react-redux";
import {useNavigate} from 'react-router-dom';

function RightMenu() {
  //const user = useSelector(state => state.user)
  //let navigate = useNavigate();

  
  return (
    <ul>
        <li><a href="/library">My Library</a></li>
      </ul>
  )
}

export default RightMenu;