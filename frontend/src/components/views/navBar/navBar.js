// import React, { useState } from 'react';
// import RightMenu from './section/rightMenu';
import './section/nav.css';
import logo from "../../images/logo_extract.png";

function NavBar() {
  //frontend/src/components/views/navBar/navBar.js
  //logo/logo_extract.png

  return (
    <nav className="menu" >
      <div className="menu__logo">
        <a href="/">Au-Dionysos</a>
        <img src={logo} alt=""></img>
      </div>
      <br/>
    </nav>
  )
}

export default NavBar;