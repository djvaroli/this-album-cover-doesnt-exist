import multiprocessing
import time
import io
import os
from pathlib import Path

import requests
from PIL import Image

from data.database import SpotifyTrack, orm


IMAGE_DIRECTORY = Path(os.path.abspath(os.path.dirname(__file__))).parent / "album_covers"
if IMAGE_DIRECTORY.exists() is False:
    IMAGE_DIRECTORY.mkdir()


def get_image_from_url(image_url: str, retry_count: int = 5) -> Image:
    """Given a URL will attempt to download the image contained in the URL.

    Args:
        image_url:
        retry_count:

    Returns:

    """

    response = requests.get(image_url, stream=True)
    count = 1
    while response.status_code != 200 and count <= retry_count:
        response = requests.get(image_url, stream=True)
        count += 1

    if "image" not in response.headers.get("content-type", ""):
        raise Exception("Unable to download image from provided URL.")

    return Image.open(io.BytesIO(response.content))


def download_track_album_covers(spotify_track: SpotifyTrack, **kwargs) -> None:
    """Fetches and saves the cover for a given Spotify track.

    Args:
        spotify_track:
        **kwargs:

    Returns:

    """
    album_cover_url = spotify_track.album_cover_url
    track_id = spotify_track.track_id

    try:
        image = get_image_from_url(album_cover_url, **kwargs)
        path = f"{IMAGE_DIRECTORY}/{track_id}.png"
        image.save(path)
    except:
        pass


if __name__ == "__main__":
    track_generator = orm.select(track for track in SpotifyTrack)

    start_time = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        print("Working.")
        completed = 0
        for _ in pool.imap_unordered(download_track_album_covers, track_generator):
            completed += 1
            print(f"Completed {completed}")

    print(f"Done!. Took {time.time() - start_time: .2f} seconds.")


