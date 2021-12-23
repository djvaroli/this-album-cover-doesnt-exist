"""
A script that takes a JSON file containing a list of Spotify album objects and extracts all the tracks in those albums.
Then stores the track data in a SQL table.
"""

from argparse import ArgumentParser
from typing import List, Dict

from tqdm import tqdm

from data_collection.database import DB, SpotifyTrack, orm
from data_collection import file_utils, spotify_utils


def store_tracks_in_db(tracks: List[dict]):
    """
    Given a list of tracks stores them in the database.
    The decorator handles committing to the database automatically.
    :param tracks:
    :return:
    """
    for track in tracks:
        try:
            db_track = SpotifyTrack(**track)
        except ValueError as e:
            pass
        except Exception as e:
            pass


def store_track_data_given_albums(source_filepath: str):
    """
    Given a file with Spotify album data, extracts all individual tracks and stores
    associated data in a MySQL database.
    :param source_filepath:
    :return:
    """

    # load the albums data from a json file
    albums: List[Dict] = file_utils.load_json(source_filepath)
    spotify_client = spotify_utils.get_spotify_client()

    for album in tqdm(albums, total=len(albums)):
        with orm.db_session:
            if SpotifyTrack.is_album_in_table(album["id"]):
                continue

            album_cover_url = album["image_url"]
            album_tracks: List[Dict] = spotify_utils.get_album_tracks(spotify_client, album["id"])
            for track in album_tracks:
                track["album_cover_url"] = album_cover_url
                track["album_name"] = album["name"]
                track["album_id"] = album["id"]

            store_tracks_in_db(album_tracks)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--source_filepath",
        type=str,
        help="Path to JSON file where to find album data",
        default="generated_files/albums.json"
    )

    args = parser.parse_args().__dict__
    store_track_data_given_albums(**args)
