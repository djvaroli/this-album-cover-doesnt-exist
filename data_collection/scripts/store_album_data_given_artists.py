from argparse import ArgumentParser

from tqdm import tqdm

import spotify_utils, file_utils


def main(source_filepath: str = "generated_files/artists.json", out_filepath: str = "generated_files/albums.json"):
    """

    :param source_filepath:
    :param out_filepath:
    :return:
    """

    artists = file_utils.load_json(source_filepath)
    print(f"Identified {len(artists)} total artists.")

    spotify_client = spotify_utils.get_spotify_client()
    albums = []
    seen_artists = set()
    for artist in tqdm(artists):
        if artist["id"] in seen_artists:
            continue
        artist_albums = spotify_utils.get_artist_albums(spotify_client, artist["id"])
        albums.extend(artist_albums)
        seen_artists.add(artist["id"])

    print(f"Identified {len(seen_artists)} unique artists.")

    file_utils.save_json(out_filepath, albums)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source_filepath", "Path to file where to find artist data.", type=str, default="generated_files/artists.json"
    )
    parser.add_argument(
        "--out_filepath", "Path to file where to save extracted album data", type=str, default="generated_files/albums.json"
    )

    args = parser.parse_args().__dict__
    main(**args)