###########
# Imports #
###########
import os
import pandas as pd
import numpy as np
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm import LightFM
# from sklearn.model_selection import train_test_split

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
train_users, test_users = np.split(users.sample(frac=1, random_state=42), [int(.8*len(users))])
bool_ratings = ratings[ratings['rating'] > 3]
train_ratings = bool_ratings[bool_ratings['user_id'].isin(train_users['user_id'].unique())]
test_ratings = bool_ratings[bool_ratings['user_id'].isin(test_users['user_id'].unique())]
# train_users, validate_users, test_users = np.split(users.sample(frac=1, random_state=42), [int(.6*len(users)), int(.8*len(users))])
# train_ratings = ratings[ratings['user_id'].isin(train_users['user_id'].unique())]
# validate_ratings = ratings[ratings['user_id'].isin(validate_users['user_id'].unique())]
# test_ratings = ratings[ratings['user_id'].isin(test_users['user_id'].unique())]

##############
# Print data #
##############

# print("===== movies =====")
# print(repr(movies))
# print("===== end movies =====")
# print("===== users =====")
# print(repr(users))
# print(repr(train_users))
# print(repr(validate_users))
# print(repr(test_users))
# print("===== end users =====")
# print("===== ratings =====")
# print(repr(ratings))
# print(repr(train_ratings))
# print(repr(validate_ratings))
# print(repr(test_ratings))
# print("===== end ratings =====")

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
# (train_interactions, weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
#                                                       for _, x in train_ratings.iterrows()))
# print(repr(train_interactions))
# (test_interactions, weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
#                                                       for _, x in test_ratings.iterrows()))
# print(repr(test_interactions))
(train_interactions, train_weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
                                                      for _, x in train_ratings.iterrows()))
(test_interactions, test_weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
                                                      for _, x in test_ratings.iterrows()))
# print("===== interactions =====")
# print(repr(interactions))
# print("===== end interactions =====")
# (train_interactions, test_interactions) = random_train_test_split(interactions, random_state=42)
# print("===== train_interactions =====")
# print(repr(train_interactions))
# print("===== end train_interactions =====")
# print("===== test_interactions =====")
# print(repr(test_interactions))
# print("===== end test_interactions =====")
user_features = dataset.build_user_features(((x['user_id'], [(x['gender'], x['age'], x['occupation'], x['zip_code'])])
                                              for _, x in users.iterrows()))
# print("===== user_features =====")
# print(repr(user_features))
# print("===== end user_features =====")
movie_features = dataset.build_item_features(((x['movie_id'], x['genre'].split('|'))
                                              for _, x in movies.iterrows()))
# print("===== movie_features =====")
# print(repr(movie_features))
# print("===== end movie_features =====")

###################
# Build the model #
###################
model = LightFM(loss='bpr')
model.fit(train_interactions, user_features=user_features, item_features=movie_features, epochs=500)

############
# Evaluate #
############
train_auc = auc_score(model,
                      train_interactions,
                      user_features=user_features,
                      item_features=movie_features).mean()
print('Training set AUC: %s' % train_auc)
test_auc = auc_score(model,
                     test_interactions,
                     user_features=user_features,
                     item_features=movie_features).mean()
print('Test set AUC: %s' % test_auc)
# train_precision = precision_at_k(model, train_interactions, user_features=user_features, item_features=movie_features).mean()
# print('Training set precision: %.2f' % train_precision)
# test_precision = precision_at_k(model, test_interactions, user_features=user_features, item_features=movie_features).mean()
# print('Test set precision: %.2f' % test_precision)

###########
# Predict #
###########
# user_id = test_users['user_id'].iloc[0]
# user_list = [user_id] * 10
# movie_list = [588, 1, 3052, 1479, 216, 10, 11, 12, 13, 14]
# print(movie_list)
# result = model.predict(user_ids=user_list, item_ids=movie_list)
# print('Prediction result: %s' % result)
# user_ratings = test_ratings[test_ratings['user_id'].isin([user_id])]
# print('Actual:\n %s' % user_ratings)
