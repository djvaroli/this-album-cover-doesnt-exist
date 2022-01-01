# Data Collection

This directory contains the logic needed to fetch the relevant Spotify
data that is then used to train the ML model that generates album covers.

### Getting Started
*NOTE*: Before attempting to use the code in this repo be aware that you will need 

* A Spotify developer account (https://developer.spotify.com/)
* A working connection to a SQL database

If both of these conditions are met you should be able to re-create the dataset I used, but one that will
be tailored to your Spotify library and hence your musical interests.

To be able to run the code follow these steps.

1. Create a `.env` file in this repo. That file should have the following variables
```dotenv
   SPOTIPY_CLIENT_SECRET=<>
   SPOTIPY_CLIENT_ID=<>
   SPOTIPY_REDIRECT_URI=<>
   MYSQL_USER=<>
   MYSQL_PASSWORD=<>
   MYSQL_HOST=<>
```

2. Make sure you have installed all the packages with `poetry install` from within the repos root directory.
 Which is the parent directory to this one.

3. Run the FastAPI app by running `uvicorn app:app --reload`

4. Open up a browser window and navigate to `http://127.0.0.1/spotify/authorize`. Which should prompt you to authorize access
to your spotify account.

5. Once that's done you are free to run any of the scripts located in the `scripts/` directory!


