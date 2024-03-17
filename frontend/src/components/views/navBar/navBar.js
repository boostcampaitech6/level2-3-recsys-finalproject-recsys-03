import React, { useState } from 'react';
import RightMenu from './section/rightMenu';
import './section/nav.css';

function NavBar() {

  return (
    <nav className="menu" >
      <div className="menu__logo">
        <a href="/">Suggestify</a>
      </div>
      <br/>
    </nav>
  )
}

export default NavBar