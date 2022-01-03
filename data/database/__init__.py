import os
import pathlib

from pony import orm
from dotenv import load_dotenv

load_dotenv()

relative_root = pathlib.Path(__file__).parent.absolute()

MYSQL_CONFIG = {
    "provider": "mysql",
    "user": os.environ["MYSQL_USER"],
    "passwd": os.environ["MYSQL_PASSWORD"],
    "host": os.environ["MYSQL_HOST"],
    "ssl_ca": os.path.join(relative_root, "certs/server-ca.cer"),
    "ssl_cert": os.path.join(relative_root, "certs/client-cert.cer"),
    "ssl_key": os.path.join(relative_root, "certs/client-key.cer"),
    "db": "spotify_data",
}

DB = orm.Database()

from .tables import *

DB.bind(**MYSQL_CONFIG)
DB.generate_mapping(create_tables=True)
