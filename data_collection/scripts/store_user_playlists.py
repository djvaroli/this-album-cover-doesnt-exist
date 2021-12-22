"""
A script that stores a list of all the playlists saved to a user's Spotify library
"""


from argparse import ArgumentParser

from data_collection import spotify_utils, file_utils


def main(out_filepath: str = "generated_files/playlists.json"):
    """
    Fetches data for all the playlists saved to a users library
    :param out_filepath:
    :return:
    """
    spotify_client = spotify_utils.get_spotify_client()
    user_playlists = spotify_utils.get_users_playlists(spotify_client)
    file_utils.save_json(out_filepath, user_playlists)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--out_filepath",
        "path to the file where to save playlist data.",
        type=str,
        default="generated_files/playlists.json"
    )
    args = parser.parse_args().__dict__
    main(**args)
