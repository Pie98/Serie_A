import pandas as pd 
import numpy as np
import pandasql as ps
import os
import re
import random 
import numpy as np
import warnings
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess_features_time_series(df_Serie_A, num_features, random_state = True):

    all_features = ['ft_goals','ft_goals_conceded','shots','shots_target', 'fouls_done','corners_obtained', 'yellows', 'reds']
    less_features = ['ft_goals','ft_goals_conceded','shots', 'fouls_done','corners_obtained', 'reds']
    few_features = ['ft_goals','ft_goals_conceded','shots', 'reds']

    if random_state:
        Train_df, Test_df, Train_labels, Test_labels = train_test_split(df_Serie_A.drop('ft_result', axis=1), df_Serie_A['ft_result'],
                                                                        test_size=0.05, random_state=42)

        Train_df, Valid_df, Train_labels, Valid_labels = train_test_split(Train_df, Train_labels, test_size=0.11, random_state=42)

    else:
        Train_df = df_Serie_A.iloc[:int(len(df_Serie_A)*0.85)]
        Valid_df = df_Serie_A.iloc[int(len(df_Serie_A)*0.85):int(len(df_Serie_A)*0.97)]
        Test_df = df_Serie_A.iloc[int(len(df_Serie_A)*0.95):]

        Train_labels = Train_df[['ft_result']]
        Valid_labels = Valid_df[['ft_result']]
        Test_labels = Test_df[['ft_result']]


    # preprocess Train dataframe
    Train_odds = Train_df[['home_win_odds','draw_odds','away_win_odds']]
    Train_teams = Train_df[['hometeam','awayteam']]

    if num_features == 'all':
        features = all_features
        print('utilizzando tutte le features')
    elif num_features == 'less':
        print('utilizzando meno features')
        features = less_features
    else:
        print('utilizzando poche features')
        features=few_features

    Train_dict_features={}

    for feature in features:
        Train_dict_features[feature] = pd.DataFrame({})
        for colonna in Train_df.columns:
            pattern = re.compile(rf'^home_{feature}_\d+$')
            if pattern.match(colonna):
                Train_dict_features[feature][colonna]=Train_df[colonna]
        for colonna in Train_df.columns:
            pattern = re.compile(rf'^away_{feature}_\d+$')
            if pattern.match(colonna):
                Train_dict_features[feature][colonna]=Train_df[colonna]

    #preprocess valid dataframe
    Valid_odds = Valid_df[['home_win_odds','draw_odds','away_win_odds']]
    Valid_teams = Valid_df[['hometeam','awayteam']]


    if num_features == 'all':
        features = all_features
        print('utilizzando tutte le features')
    elif num_features == 'less':
        print('utilizzando meno features')
        features = less_features
    else:
        print('utilizzando poche features')
        features=few_features

    Valid_dict_features={}

    for feature in features:
        Valid_dict_features[feature] = pd.DataFrame({})
        for colonna in Valid_df.columns:
            pattern = re.compile(rf'^home_{feature}_\d+$')
            if pattern.match(colonna):
                Valid_dict_features[feature][colonna]=Valid_df[colonna]
        for colonna in Valid_df.columns:
            pattern = re.compile(rf'^away_{feature}_\d+$')
            if pattern.match(colonna):
                Valid_dict_features[feature][colonna]=Valid_df[colonna]

    # preprocess test dataframe

    Test_odds = Test_df[['home_win_odds','draw_odds','away_win_odds']]
    Test_teams = Test_df[['hometeam','awayteam']]

    if num_features == 'all':
        features = all_features
        print('utilizzando tutte le features')
    elif num_features == 'less':
        print('utilizzando meno features')
        features = less_features
    else:
        print('utilizzando poche features')
        features=few_features

    Test_dict_features={}

    for feature in features:
        Test_dict_features[feature] = pd.DataFrame({})
        for colonna in Test_df.columns:
            pattern = re.compile(rf'^home_{feature}_\d+$')
            if pattern.match(colonna):
                Test_dict_features[feature][colonna]=Test_df[colonna]
        for colonna in Test_df.columns:
            pattern = re.compile(rf'^away_{feature}_\d+$')
            if pattern.match(colonna):
                Test_dict_features[feature][colonna]=Test_df[colonna]

    #encoding teams
    teams_transf = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['hometeam','awayteam']),
    sparse_threshold=0  
    )

    teams_transf.fit(Train_teams)

    Train_teams_encoded = teams_transf.transform(Train_teams)
    Valid_teams_encoded = teams_transf.transform(Valid_teams)
    Test_teams_encoded = teams_transf.transform(Test_teams)

    if random_state == False:
        #encoding labels
        label_transf = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['ft_result']),
        sparse_threshold=0  
        )

        label_transf.fit(Train_labels)
        Train_labels_encoded = label_transf.transform(Train_labels)
        Valid_labels_encoded = label_transf.transform(Valid_labels)
        Test_labels_encoded = label_transf.transform(Test_labels)

    else:
        # Crea un OrdinalEncoder
        ordinal_encoder = OneHotEncoder(sparse=False)
        # Addestra l'OrdinalEncoder su Train_labels e applica la trasformazione
        Train_labels_encoded = ordinal_encoder.fit(np.array(Train_labels).reshape(-1, 1))

        Train_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Train_labels).reshape(-1, 1)))
        Valid_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Valid_labels).reshape(-1, 1)))
        Test_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Test_labels).reshape(-1, 1)))  

    #encoding numerical features
    Train_dict_features_norm = Train_dict_features.copy()
    Valid_dict_features_norm = Valid_dict_features.copy()
    Test_dict_features_norm = Test_dict_features.copy()

    for feature in list(Train_dict_features.keys()):
        feature_list = Train_dict_features_norm[feature].columns
        numerical_transf = make_column_transformer(
            (MinMaxScaler(), feature_list), 
            sparse_threshold=0  
        )
        numerical_transf.fit(Train_dict_features[feature])
        Train_dict_features_norm[feature] = numerical_transf.transform(Train_dict_features_norm[feature])
        Valid_dict_features_norm[feature] = numerical_transf.transform(Valid_dict_features_norm[feature])
        Test_dict_features_norm[feature] = numerical_transf.transform(Test_dict_features_norm[feature])

    return (Train_teams_encoded, Valid_teams_encoded, Test_teams_encoded, Train_labels_encoded, Valid_labels_encoded, Test_labels_encoded, 
            Train_dict_features_norm, Valid_dict_features_norm, Test_dict_features_norm, Train_teams, Valid_teams, Test_teams, Train_labels, Valid_labels, Test_labels, 
            Train_dict_features, Valid_dict_features, Test_dict_features, Train_odds, Valid_odds, Test_odds)



# -------------------------------------------#

def create_fast_preprocessing_ts(Train_teams_encoded, Train_dict_features_norm, Train_labels_encoded,Valid_teams_encoded, Valid_dict_features_norm,
                                 Valid_labels_encoded,Test_teams_encoded, Test_dict_features_norm,Test_labels_encoded ):
    #creo i fast preprocessing datasets
    Dataset_train_norm = tf.data.Dataset.from_tensor_slices(Train_dict_features_norm[list(Train_dict_features_norm.keys())[0]])
    for feature in list(Train_dict_features_norm.keys())[1:]:
        temp_dataset = tf.data.Dataset.from_tensor_slices(Train_dict_features_norm[feature])
        Dataset_train_norm = tf.data.Dataset.zip((Dataset_train_norm, temp_dataset))
    Train_labels_encoded = tf.data.Dataset.from_tensor_slices(Train_labels_encoded) # make labels
    Dataset_train_norm = tf.data.Dataset.zip((Dataset_train_norm, Train_labels_encoded))

    #creo un array con le features concatenate
    Dataset_Valid_norm = tf.data.Dataset.from_tensor_slices(Valid_dict_features_norm[list(Valid_dict_features_norm.keys())[0]])
    for feature in list(Valid_dict_features_norm.keys())[1:]:
        temp_dataset = tf.data.Dataset.from_tensor_slices(Valid_dict_features_norm[feature])
        Dataset_Valid_norm = tf.data.Dataset.zip((Dataset_Valid_norm, temp_dataset))
    Valid_labels_encoded = tf.data.Dataset.from_tensor_slices(Valid_labels_encoded) # make labels
    Dataset_Valid_norm = tf.data.Dataset.zip((Dataset_Valid_norm, Valid_labels_encoded))

    #creo un array con le features concatenate
    Dataset_Test_norm = tf.data.Dataset.from_tensor_slices(Test_dict_features_norm[list(Test_dict_features_norm.keys())[0]])
    for feature in list(Test_dict_features_norm.keys())[1:]:
        temp_dataset = tf.data.Dataset.from_tensor_slices(Test_dict_features_norm[feature])
        Dataset_Test_norm = tf.data.Dataset.zip((Dataset_Test_norm, temp_dataset))
    Test_labels_encoded = tf.data.Dataset.from_tensor_slices(Test_labels_encoded) # make labels
    Dataset_Test_norm = tf.data.Dataset.zip((Dataset_Test_norm, Test_labels_encoded))

    Dataset_train_norm = Dataset_train_norm.batch(32).prefetch(tf.data.AUTOTUNE) #Autotune è per dirgli di prefetchare tanti dati quanti può
    Dataset_valid_norm = Dataset_Valid_norm.batch(32).prefetch(tf.data.AUTOTUNE)
    Dataset_Test_norm = Dataset_Test_norm.batch(32).prefetch(tf.data.AUTOTUNE)

    return Dataset_train_norm, Dataset_valid_norm, Dataset_Test_norm