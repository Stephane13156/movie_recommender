"""
In this script we define functions for the recommender web
application
"""

import pandas as pd
import numpy as np

#from utils import MOVIES, nmf_model, cos_sim_model

def recommend_nmf(query, model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    df_movies = pd.read_csv('./data/movies_mod.csv',sep=',')

    
    # construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(query, columns=df_movies['title'], index=['new_user'])
    new_user_dataframe
    
    # filling missing values NaN with 0
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)
    
    # create P Matrix for new user
    P_new_user_matrix = model.transform(new_user_dataframe_imputed)
    
    # create P Matrix DataFrame
    P_new_user = pd.DataFrame(P_new_user_matrix, 
                         columns = model.get_feature_names_out(),
                         index = ['new_user'])
    
    # create user-feature matrix P for new-user
    Q_matrix = model.components_
    Q = pd.DataFrame(Q_matrix, columns=df_movies['title'], index=model.get_feature_names_out())
    R_hat_new_user_matrix = np.dot(P_new_user,Q)
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=df_movies['title'],
                         index = ['new_user'])
    
    # Remove seen & rated movies
    R_hat_new_filtered = R_hat_new_user.drop(query.keys(),axis=1)
   
    # scoring
    
        # calculate the score with the NMF model
    R_hat_new_filtered.T.sort_values(by=["new_user"],ascending=False).index.tolist()
    
    
    # ranking
    
        # return the top-k highest rated movie ids or titles
    ranked =  R_hat_new_filtered.T.sort_values(by=["new_user"],ascending=False).index.tolist()
    
    recommended = ranked[:k]
    return recommended

def recommend_neighborhood(query, model, ratings, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    df_movies = pd.read_csv('./data/movies_mod.csv',sep=',')
    R = pd.read_csv('./data/R_Matrix.csv',sep=',')

    # 1. candiate generation   
    # construct a user vector
    new_user_dataframe =  pd.DataFrame(query, columns=df_movies['title'], index=['new_user'])
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)
    
   
    # 2. scoring
    
    # find n neighbors
    # calculates the distances to all other users in the data!
    similarity_scores, neighbor_ids = model.kneighbors(
    new_user_dataframe_imputed,
    n_neighbors=5,
    return_distance=True
    )
    # sklearn returns a list of predictions
    # extract the first and only value of the list
    neighbors_df = pd.DataFrame(data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]})
    # only look at ratings for users that are similar!
    neighborhood = R.iloc[neighbor_ids[0]]
    # filter out movies allready seen by the user
    neighborhood_filtered = neighborhood.drop(query.keys(),axis=1)
    # calculate the summed up rating for each movie
    # summing up introduces a bias for popular movies
    # averaging introduces bias for movies only seen by few users in the neighboorhood

    df_score = neighborhood_filtered.sum()
    
    # 3. ranking
    df_score_ranked = df_score.sort_values(ascending=False).index.tolist()
    
    # return the top-k highst rated movie ids or titles
    recommended = df_score_ranked[:k]
    
    return recommended


#def random_recommender(k=2):
#    if k > len(MOVIES):
#        print (f"Hey, you exceeded the number of available movies {len(MOVIES)}")
#        return []
#    else:
#        random.shuffle(MOVIES)
#        top_k = MOVIES[:k]
#        return top_k

#if __name__ == "__main__":
#    top2 = random_recommender()
#    print(top2)