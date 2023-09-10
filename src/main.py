###########
# Imports #
###########
import pandas as pd
import os
from lightfm.data import Dataset
from lightfm import LightFM

#########
# Setup #
#########
cwd = os.getcwd()

#############
# Load data #
#############
movies = pd.read_csv(os.path.join(cwd, 'ml-1m', 'movies.dat'),
                     delimiter='::', engine= 'python', header=None,
                     names=['movie_id', 'movie_name', 'genre'], encoding="ISO-8859-1")
print(movies.head())
users = pd.read_csv(os.path.join(cwd, 'ml-1m', 'users.dat'),
                    delimiter='::', engine='python', header=None,
                    names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
print(users.head())
ratings = pd.read_csv(os.path.join(cwd, 'ml-1m', 'ratings.dat'),
                      delimiter='::', engine='python', header=None,
                      names=['user_id', 'movie_id', 'rating', 'time'])
print(ratings.head())

###################
# Prepare dataset #
###################
dataset = Dataset()
dataset.fit(users['user_id'],
            movies['movie_id'],
            ((x['gender'], x['age'], x['occupation'], x['zip_code']) for _, x in users.iterrows()),
            ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
num_users, num_movies = dataset.interactions_shape()
print('Num users: {}, num movies: {}.'.format(num_users, num_movies))
(interactions, weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
                                                      for _, x in ratings.iterrows()))
print(repr(interactions))
user_features = dataset.build_user_features(((x['user_id'], [(x['gender'], x['age'], x['occupation'], x['zip_code'])])
                                              for _, x in users.iterrows()))
print(repr(user_features))
movie_features = dataset.build_item_features(((x['movie_id'], x['genre'].split('|'))
                                              for _, x in movies.iterrows()))
print(repr(movie_features))

###################
# Build the model #
###################
model = LightFM(loss='bpr')
model.fit(interactions, user_features=user_features, item_features=movie_features)
