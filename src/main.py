###########
# Imports #
###########
import os
import pandas as pd
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from sklearn.model_selection import train_test_split

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
users = pd.read_csv(os.path.join(cwd, 'ml-1m', 'users.dat'),
                    delimiter='::', engine='python', header=None,
                    names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])
ratings = pd.read_csv(os.path.join(cwd, 'ml-1m', 'ratings.dat'),
                      delimiter='::', engine='python', header=None,
                      names=['user_id', 'movie_id', 'rating', 'time'])

################################################
# Split the data to train, test and validation #
################################################
train_users, validate_users, test_users = np.split(users.sample(frac=1, random_state=42), [int(.6*len(users)), int(.8*len(users))])
train_ratings = ratings[ratings['user_id'].isin(train_users['user_id'].unique())]
validate_ratings = ratings[ratings['user_id'].isin(validate_users['user_id'].unique())]
test_ratings = ratings[ratings['user_id'].isin(test_users['user_id'].unique())]

##############
# Print data #
##############

print("===== movies =====")
print(repr(movies))
print("===== end movies =====")
print("===== users =====")
print(repr(users))
print(repr(train_users))
print(repr(validate_users))
print(repr(test_users))
print("===== end users =====")
print("===== ratings =====")
print(repr(ratings))
print(repr(train_ratings))
print(repr(validate_ratings))
print(repr(test_ratings))
print("===== end ratings =====")

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
                                              for _, x in train_users.iterrows()))
print(repr(user_features))
movie_features = dataset.build_item_features(((x['movie_id'], x['genre'].split('|'))
                                              for _, x in movies.iterrows()))
print(repr(movie_features))

###################
# Build the model #
###################
model = LightFM(loss='bpr')
model.fit(interactions, user_features=user_features, item_features=movie_features, epochs=30)
