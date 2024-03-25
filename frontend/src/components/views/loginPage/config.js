export const CLIENT_ID = 'a6034aad50b54280b0c197191b51d301'
export const SPOTIFY_AUTHORIZE_ENDPOINT = 'https://accounts.spotify.com/authorize'
export const REDIRECT_URL = "https://au-dionysos.com/service"
const SPACE_DELIMITER = "%20"
const SCOPES = [
    "user-top-read", 
    "playlist-read-private", 
    "user-read-playback-state", 
    "user-modify-playback-state", 
    "streaming", 
    "app-remote-control", 
    "user-read-private", 
    "user-read-email", 
    "playlist-modify-public"]
export const SCOPES_PARAM = SCOPES.join(SPACE_DELIMITER)