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
        <Routes>
            <Route exact path='/' element={<Spotify_login/>} />
            <Route exact path='/service' element={<Service/>} />
        </Routes>
      </div>
    </Router>
  );
}
