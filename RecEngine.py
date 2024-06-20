import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit.string_util import clean_text

# from utils import clean
shows = pd.read_csv('TMDB_tv_dataset_v3.csv')
shows.drop_duplicates(inplace=True)
shows.dropna(inplace=True)
print(shows.columns)
shows = shows[['id',
               'name',
               'overview',
               'genres',
               'number_of_seasons',
               'number_of_episodes',
               'first_air_date',
               'last_air_date',
               'networks',
               'vote_average',
               'languages']]
shows['tags'] = shows['overview'] + shows['genres']
new_data = shows.drop(columns=['overview', 'genres'])
new_data['clean_tags'] = new_data['tags'].apply(clean_text)
cv = CountVectorizer(max_features=75000, stop_words='english')
vectorized_data = cv.fit_transform(new_data['clean_tags']).toarray()
similarity = cosine_similarity(vectorized_data)
new_data.info()


def recommend_movies():
    pass


def clean_text():
    pass


shows.describe()
print(shows['tags'].head())
