###########
# Imports #
###########
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
users['gender'] = users['gender'].replace(['M', 'F'], [0, 1])
ratings = pd.read_csv(os.path.join(cwd, 'ml-1m', 'ratings.dat'),
                      delimiter='::', engine='python', header=None,
                      names=['user_id', 'movie_id', 'rating', 'time'])

################################################
# Split the data to train, test and validation #
################################################
train_users, test_users = np.split(users.sample(frac=1, random_state=42), [int(.8*len(users))])
train_ratings = ratings[ratings['user_id'].isin(train_users['user_id'].unique())]
test_ratings = ratings[ratings['user_id'].isin(test_users['user_id'].unique())]
# train_ratings = train_ratings[train_ratings['rating'].isin([5])]
# test_ratings = test_ratings[test_ratings['rating'].isin([5])]
# print('ratings:\n{}'.format(repr(ratings)))
# print('train_ratings:\n{}'.format(repr(train_ratings)))
# print('test_ratings:\n{}'.format(repr(test_ratings)))
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
            # ((x['gender'], x['age'], x['occupation'], x['zip_code']) for _, x in users.iterrows()),
            ["Gender", "Age", "Occupation"],
            ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])
num_users, num_movies = dataset.interactions_shape()
print('Num users: {}, num movies: {}.'.format(num_users, num_movies))
# (train_interactions, weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
#                                                       for _, x in train_ratings.iterrows()))
# print(repr(train_interactions))
# (test_interactions, weights) = dataset.build_interactions(((x['user_id'], x['movie_id'])
#                                                       for _, x in test_ratings.iterrows()))
# print(repr(test_interactions))
(train_interactions, train_weights) = dataset.build_interactions(((x['user_id'], x['movie_id'], x['rating'])
                                                      for _, x in train_ratings.iterrows()))
(test_interactions, test_weights) = dataset.build_interactions(((x['user_id'], x['movie_id'], x['rating'])
                                                      for _, x in test_ratings.iterrows()))
# print("===== interactions =====")
# print(repr(interactions))
# print("===== end interactions =====")
# (train_interactions, test_interactions) = random_train_test_split(interactions, random_state=42)
# print("===== train_interactions =====")
# print('train_interactions:\n{}'.format(train_interactions))
# print('train_weights:\n{}'.format(train_weights))
# print("===== end train_interactions =====")
# print("===== test_interactions =====")
# print('test_interactions:\n{}'.format(test_interactions))
# print("===== end test_interactions =====")
user_features = dataset.build_user_features(((x['user_id'], {"Gender": float(x['gender']), "Age": float(x['age']), "Occupation": float(x['occupation'])})
                                              for _, x in users.iterrows()), normalize=False)
# print('user_features:\n{}'.format(user_features))
# print("===== user_features =====")
# print(repr(user_features))
# print("===== end user_features =====")
movie_features = dataset.build_item_features(((x['movie_id'], x['genre'].split('|'))
                                              for _, x in movies.iterrows()), normalize=False)
# print('movie_features:\n{}'.format(movie_features))
# print(movie_features.shape[0])
# print("===== movie_features =====")
# print(repr(movie_features))
# print("===== end movie_features =====")

###################
# Build the model #
###################
warp_model = LightFM(loss='warp')
# bpr_model = LightFM(loss='bpr')
warp_model.fit(train_weights, user_features=user_features, item_features=movie_features, epochs=10)
# bpr_model.fit(train_interactions, user_features=user_features, item_features=movie_features, epochs=10)
# train_auc = auc_score(warp_model,
#                       test_interactions,
#                       train_interactions,
#                       user_features=user_features,
#                       item_features=movie_features).mean()
# print('AUC warp_model: %s' % train_auc)
# test_auc = auc_score(warp_model,
#                      test_interactions,
#                      user_features=user_features,
#                      item_features=movie_features).mean()
# print('Test set AUC warp_model: %s' % test_auc)
# train_auc = auc_score(bpr_model,
#                       test_interactions,
#                       train_interactions,
#                       user_features=user_features,
#                       item_features=movie_features).mean()
# print('AUC bpr_model: %s' % train_auc)
# test_auc = auc_score(bpr_model,
#                      test_interactions,
#                      user_features=user_features,
#                      item_features=movie_features).mean()
# print('Test set AUC bpr_model: %s' % test_auc)
# warp_auc = []
# bpr_auc = []
# for epoch in range(10):
#     print('Epoch: {}'.format(epoch))
#     warp_model.fit_partial(train_interactions, user_features=user_features, item_features=movie_features, epochs=1)
#     bpr_model.fit_partial(train_interactions, user_features=user_features, item_features=movie_features, epochs=1)
#     warp_auc.append(auc_score(warp_model, test_interactions, train_interactions, user_features=user_features, item_features=movie_features).mean())
#     bpr_auc.append(auc_score(bpr_model, test_interactions, train_interactions, user_features=user_features, item_features=movie_features).mean())
#     print('warp_auc: {}\nbpr_auc: {}'.format(warp_auc, bpr_auc))
# x = np.arange(10)
# plt.plot(x, np.array(warp_auc))
# plt.plot(x, np.array(bpr_auc))
# plt.legend(['WARP AUC', 'BPR AUC'], loc='upper right')
# plt.show()

############
# Evaluate #
############
train_auc = auc_score(warp_model,
                      test_weights,
                      train_weights,
                      user_features=user_features,
                      item_features=movie_features).mean()
print('AUC warp_model: %s' % train_auc)
train_precision = precision_at_k(warp_model, test_weights, train_weights, k=5, user_features=user_features, item_features=movie_features).mean()
print('Precision: %.2f' % train_precision)

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
list_of_unique_test_users = test_users['user_id'].unique()
predictions = []
for i in range(len(list_of_unique_test_users)):
    predictions.append(warp_model.predict(np.full(3883, list_of_unique_test_users[i]-1), np.arange(0, 3883), item_features=movie_features, user_features=user_features))
print(predictions)
