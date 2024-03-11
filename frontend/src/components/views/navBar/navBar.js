import React, { useState } from 'react';
import RightMenu from './section/rightMenu';
import './section/nav.css';

function NavBar() {

  return (
    <nav className="menu" >
      <div className="menu__logo">
        <a href="/service">Suggestify</a>
      </div>
      <div className="menu_rigth">
          <RightMenu mode="horizontal" />
      </div>
      <br/>
      <br/>
      <hr/>
    </nav>
  )
}

export default NavBar