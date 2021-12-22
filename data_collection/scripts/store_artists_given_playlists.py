"""
A script that takes a JSON file containing a list of Spotify playlist objects and extracts all the
artists that are present in the playlist. Then stores the artist data in a JSON file.
"""

from tqdm import tqdm

from data_collection import file_utils, spotify_utils


def get_artists_from_playlist(source_filepath: str = "data/playlists.json", out_filepath: str = "data/artists.json"):
    """
    Given a file with a list of playlists fetches and saves all the artists in the playlists
    :param source_filepath:
    :param out_filepath:
    :return:
    """
    playlists = file_utils.load_json(source_filepath)
    spotify_client = spotify_utils.get_spotify_client()
    artists = []
    for playlist in tqdm(playlists):
        playlist_artists = spotify_utils.get_playlist_artists(spotify_client, playlist["id"])
        artists.extend(playlist_artists)

    file_utils.save_json(out_filepath, artists)


if __name__ == '__main__':
    get_artists_from_playlist()