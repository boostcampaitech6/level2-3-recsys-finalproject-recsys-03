import React, { useState } from 'react';
import RightMenu from './section/rightMenu';
import './section/nav.css';

function NavBar() {
  //frontend/src/components/views/navBar/navBar.js
  //logo/logo_extract.png

  return (
    <nav className="menu" >
      <div className="menu__logo">
        <a href="/">Au-Dionysos</a>
        <img src='./logo_extract.png'></img>
      </div>
      <br/>
    </nav>
  )
}

export default NavBar