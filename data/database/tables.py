
from . import DB, orm


class SpotifyTrack(DB.Entity):
    """
    Table that contains the data related to spotify tracks.
    """
    track_id = orm.PrimaryKey(str, auto=False)
    track_name = orm.Required(str)
    album_name = orm.Required(str)
    album_id = orm.Required(str)
    album_cover_url = orm.Required(str)

    @classmethod
    def is_album_in_table(cls, album_id: str):
        """
        Checks whether or not the tracks from a specified album are already in the table in the database
        :param album_id:  the unique identifier of the Spotify album.
        :return:
        """

        # if there's at least one track matching the album id, we consider that the whole album is in the Table
        q = cls.select(album_id=album_id)
        return len(q[:1]) > 0


