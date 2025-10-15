import numpy as np
import pandas as pd
import os

json_path = "./games.json"

df = pd.read_json(json_path)
df = df.T
df.head()

column_name_dict = {
"name": "Name",
"release_date": "Release date",
"required_age": "Required age",
"price": "Price",
"dlc_count": "DLC count",
"detailed_description": "Detailed description",
"about_the_game": "About the game",
"short_description": "Short description",
"reviews": "Reviews",
"header_image": "Header image",
"website": "Website",
"support_url": "Support url",
"support_email": "Support email",
"windows": "Windows",
"mac": "Mac",
"linux": "Linux",
"metacritic_score": "Metacritic score",
"metacritic_url": "Metacritic url",
"achievements": "Achievements",
"recommendations": "Recommendations",
"notes": "Notes",
"supported_languages": "Supported languages",
"full_audio_languages": "Full audio languages",
"packages": "Packages",
"developers": "Developers",
"publishers": "Publishers",
"categories": "Categories",
"genres": "Genres",
"screenshots": "Screenshots",
"movies": "Movies",
"user_score": "User score",
"score_rank": "Score rank",
"positive": "Positive",
"negative": "Negative",
"estimated_owners": "Estimated owners",
"average_playtime_forever": "Average playtime forever",
"average_playtime_2weeks": "Average playtime two weeks",
"median_playtime_forever": "Median playtime forever",
"median_playtime_2weeks": "Median playtime two weeks",
"peak_ccu": "Peak CCU",
"tags": "Tags"
}

def convert_dict_to_string(dict_object):
    # Used for converting the .json data into the format used in the .csv file
    # i.e. dict of "Tag: tag_id" into string with comma-separated tags
    if len(dict_object) == 0:
        return np.nan
    key_list = list(dict_object.keys())
    keys_string = ",".join(key_list)
    return keys_string

def read_convert_json_dataset():
    _df = pd.read_json(json_path)
    _df = _df.T
    _df['AppID'] = _df.index
    _df.rename(columns=column_name_dict, inplace=True)

    # Convert the dict/array columns
    for col in _df.columns.values:
        if isinstance(_df.loc[546560][col], dict):
            _df[col] = _df[col].apply(lambda entries: convert_dict_to_string(entries))
        if isinstance(_df.loc[546560][col], list):
            if isinstance(_df.loc[546560][col][0], str): # Avoids trying to convert the "Packages" column
                _df[col] = _df[col].apply(lambda entries: ",".join(entries))
    _df = _df.reset_index().set_index("AppID")
    return _df

df = read_convert_json_dataset()
df.info()
df.head()
df["Tags"].head()
df["Supported languages"].head()
df.to_csv("converted.csv") # Saving the converted dataframe
converted = pd.read_csv("converted.csv") # Reading it back
converted.head()