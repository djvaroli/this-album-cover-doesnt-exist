"""
Code for utility functions for interaction with a user's Spotify account and its data
"""
from typing import List, Dict

import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()


def get_spotify_client(
        scope: str = "user-read-private user-top-read playlist-read-private",
        *args,
        **kwargs
) -> Spotify:
    """
    Returns an instance of spotify.Spotify with permission to access specified scopes
    :param scope:
    :return:
    """
    return spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope), *args, **kwargs)


def get_users_playlists(spotify_client: Spotify, *args, **kwargs) -> List[Dict]:
    """
    Returns a list of a user's playlists.
    All positional and keyword arguments will be passed to Spotify.current_user_playlists function
    :param spotify_client:
    :param args:
    :param kwargs:
    :return:
    """
    playlists = spotify_client.current_user_playlists(*args, **kwargs)["items"]
    relevant_playlist_data = []  # playlist data comes with a lot of fields we don't need
    return [
        {"name": playlist["name"], "id": playlist["id"]} for playlist in playlists
    ]


def get_album_tracks(
        spotify_client: Spotify,
        album_id: str
) -> List[Dict]:
    """
    Given an album id returns a list of tracks
    :param spotify_client:
    :param album_id:
    :return:
    """
    album_tracks = spotify_client.album_tracks(album_id)["items"]
    return [
        {"track_name": track["name"], "track_id": track["id"]} for track in album_tracks
    ]


def get_artist_albums(
        spotify_client: Spotify,
        artist_id: str,
        limit: int = 50
) -> List[Dict]:
    """
    Returns a list of Albums by the specified artist
    :param spotify_client:
    :param artist_id:
    :param limit
    :return:
    """

    artist_albums = spotify_client.artist_albums(artist_id, limit=limit)["items"]
    return [
        {
            "name": album["name"],
            "id": album["id"],
            "image_url": album["images"][0]["url"] if album["images"] else "",
        } for album in artist_albums
    ]


def get_playlist_tracks(
        spotify_client: Spotify,
        playlist_id: str,
        *args,
        **kwargs
) -> List[Dict]:
    """
    Given an id of Spotify playlist, returns a list of tracks in that playlist.
    All positional and keyword arguments will be passed on to Spotify.playlist_items (see documentation for usage)
    :param spotify_client:
    :param playlist_id:
    :return:
    """

    playlist_tracks = spotify_client.playlist_items(playlist_id=playlist_id, *args, **kwargs)["items"]
    return [
        {"name": track["name"], "id": track["id"]} for track in playlist_tracks
    ]


def get_playlist_artists(
        spotify_client: Spotify,
        playlist_id: str,
        *args,
        **kwargs
) -> List[Dict]:
    """
    Given an id of a Spotify playlist returns a list of artists in that playlist.
    :param spotify_client:
    :param playlist_id:
    :return:
    """
    playlist_tracks = spotify_client.playlist_items(playlist_id=playlist_id, *args, **kwargs)["items"]
    playlist_artists = []
    seen_artists = set()
    for track in playlist_tracks:
        for artist in track["track"]["artists"]:
            artist_name = artist["name"]
            artist_id = artist["id"]
            if artist_id in seen_artists:
                continue
            seen_artists.add(artist_id)
            playlist_artists.append(dict(id=artist_id, name=artist_name))

    return playlist_artists


def get_playlist_albums(
        spotify_client: Spotify,
        playlist_id: str,
        *args,
        **kwargs
) -> List[Dict]:
    """
    Given an id of a Spotify playlist returns a list of albums in that playlist.
    :param spotify_client:
    :param playlist_id:
    :return:
    """
    playlist_tracks = spotify_client.playlist_items(playlist_id=playlist_id, *args, **kwargs)["items"]
    playlist_albums = []
    seen_albums = set()
    for track in playlist_tracks:
        album_name = track["track"]["album"]["name"]
        album_id = track["track"]["album"]["id"]
        if album_id in seen_albums:
            continue

        seen_albums.add(album_id)
        playlist_albums.append(dict(id=album_id, name=album_name))

    return playlist_albums

