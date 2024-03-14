import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,

} from "react-router-dom";

import Spotify_login from "./components/views/loginPage/spotify_login";
import Service from "./components/views/servicePage/service";


export default function BasicExample() {
  return (
    <Router>
      <div>
          <link rel="preconnect" href="https://fonts.googleapis.com"/>
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
          <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@100..900&display=swap" rel="stylesheet"/>
        <Routes>
            <Route exact path='/' element={<Spotify_login/>} />
            <Route exact path='/service' element={<Service/>} />
        </Routes>
      </div>
    </Router>
  );
}
