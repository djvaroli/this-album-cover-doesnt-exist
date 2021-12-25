"""
A script that takes a JSON file containing a list of Spotify album objects and extracts all the tracks in those albums.
Then stores the track data in a SQL table.
"""

from argparse import ArgumentParser
from typing import List, Dict
import random

from tqdm import tqdm

from data_collection.database import DB, SpotifyTrack, orm
from data_collection import utils, spotify_utils


@orm.db_session
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


@orm.db_session
def get_tracks_for_album_batch(album_batch: List[dict]) -> List[dict]:
    """
    Gets tracks for a batch of albums
    :param album_batch:
    :return:
    """
    album_batch_tracks = []

    # renew the spotify client at every batch
    spotify_client = spotify_utils.get_spotify_client(status_retries=5, requests_timeout=10)
    for album in album_batch:

        # first check if any tracks from the album are already in the database
        if SpotifyTrack.is_album_in_table(album["id"]):
            continue

        album_cover_url = album["image_url"]
        album_tracks: List[Dict] = spotify_utils.get_album_tracks(spotify_client, album["id"])

        for track in album_tracks:
            track["album_cover_url"] = album_cover_url
            track["album_name"] = album["name"]
            track["album_id"] = album["id"]

        album_batch_tracks.extend(album_tracks)

    return album_batch_tracks


def store_track_data_given_albums(source_filepath: str):
    """
    Given a file with Spotify album data, extracts all individual tracks and stores
    associated data in a MySQL database.
    :param source_filepath:
    :return:
    """

    # load the albums data from a json file
    albums: List[Dict] = utils.load_json(source_filepath)
    random.shuffle(albums)
    album_batches = utils.list_into_batches(albums, batch_size=20)

    for album_batch in tqdm(album_batches, total=len(album_batches)):
        album_batch_tracks = get_tracks_for_album_batch(album_batch)
        store_tracks_in_db(album_batch_tracks)


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
