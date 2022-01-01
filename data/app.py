from fastapi import FastAPI

from dotenv import load_dotenv

import spotify_utils

load_dotenv()
app = FastAPI()


@app.get("/")
def home():
    """
    Home page of the application
    :return:
    """
    return "Please navigate to /spotify/authorize to authorize access to Spotify data."


@app.get("/spotify/authorize")
def authorize():
    """
    Navigates the user to authorize access to their Spotify account with the specified scopes
    :return:
    """

    spotify_client = spotify_utils.get_spotify_client()
    return {"authorization_status": "success"}


@app.get("/callback")
def authorization_callback():
    """
    Endpoint that user is redirected to once authorization step is completed
    :return:
    """

    return

