from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np 
import pandas as pd 

def preprocess_columns(dataframe,numero_colonne,giorni_cumulativi, oversample=False):
    all_columns = ['giornata', 'stagione','hometeam', 'awayteam','home_total_points','home_result', 'away_total_points', f'home_last_{giorni_cumulativi}_days_ft_goals',
               f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ht_goals',f'away_last_{giorni_cumulativi}_days_ht_goals', f'away_last_{giorni_cumulativi}_days_shots',
       f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',
       f'home_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots_target',
       f'away_last_{giorni_cumulativi}_days_shots_target',f'home_last_{giorni_cumulativi}_days_fouls_done',f'away_last_{giorni_cumulativi}_days_fouls_done',f'home_last_{giorni_cumulativi}_days_corners_obtained',
       f'away_last_{giorni_cumulativi}_days_corners_obtained',f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows',
       f'home_last_{giorni_cumulativi}_days_reds', f'away_last_{giorni_cumulativi}_days_reds']

    less_columns = ['stagione','hometeam', 'awayteam','home_total_points','home_result', 'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',
       f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'home_last_{giorni_cumulativi}_days_shots',
       f'away_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows']

    few_columns = ['hometeam', 'awayteam','home_total_points','home_result', 'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',
               f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',
               f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'home_last_{giorni_cumulativi}_days_shots',
               f'away_last_{giorni_cumulativi}_days_shots']


    if numero_colonne == 'all':
       colonne = all_columns
       print('utilizzando tutte le features')
    elif numero_colonne == 'less':
       print('utilizzando meno features')
       colonne = less_columns
    else:
       print('utilizzando poche features')
       colonne=few_columns


    for day in range(giorni_cumulativi):
       colonne.append(f'home_result_{day+1}')
       colonne.append(f'away_result_{day+1}')

    df_stats_serie_A = dataframe[colonne]   

    #creo il train valid test
    X_train_df, X_test_df, Train_labels, Test_labels = train_test_split(df_stats_serie_A.drop(f'home_result', axis=1), df_stats_serie_A[f'home_result'],
                                                                     test_size=0.05, random_state=42)

    X_train_df, X_valid_df, Train_labels, Valid_labels = train_test_split(X_train_df, Train_labels, test_size=0.11, random_state=42)

    if oversample: #questo permette di fare l'oversampling di cui si parlava prima
      ros = RandomOverSampler()
      X_train_df, Train_labels = ros.fit_resample(X_train_df, Train_labels)

    #print(f'lunghezza dataframe train {len(X_train_df)}, \n lunghezza dataframe valid {len(X_valid_df)}, \n lunghezza dataframe test {len(X_test_df)}')

    all_numerical_columns= [f'home_total_points',f'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ht_goals',
       f'away_last_{giorni_cumulativi}_days_ht_goals', f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',
       f'home_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots_target',f'away_last_{giorni_cumulativi}_days_shots',f'away_last_{giorni_cumulativi}_days_shots_target',
       f'home_last_{giorni_cumulativi}_days_fouls_done',f'away_last_{giorni_cumulativi}_days_fouls_done',f'home_last_{giorni_cumulativi}_days_corners_obtained',
       f'away_last_{giorni_cumulativi}_days_corners_obtained',f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows',f'home_last_{giorni_cumulativi}_days_reds', 
       f'away_last_{giorni_cumulativi}_days_reds']

    less_numerical_columns = [f'home_total_points',f'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',
                            f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots',
                            f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows']

    few_numerical_columns = [f'home_total_points',f'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',
                            f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots']


    all_categorical_columns = ['giornata', 'stagione',f'hometeam', f'awayteam']
    less_categorical_columns = ['stagione',f'hometeam', f'awayteam']
    few_categorical_columns = [f'hometeam', f'awayteam']

    if numero_colonne == 'all':
        categorical_columns = all_categorical_columns
        numerical_columns = all_numerical_columns
    elif numero_colonne == 'less':
        categorical_columns = less_categorical_columns
        numerical_columns = less_numerical_columns
    else:
        categorical_columns=few_categorical_columns
        numerical_columns = few_numerical_columns


    for day in range(giorni_cumulativi):
        categorical_columns.append(f'home_result_{day+1}')
        categorical_columns.append(f'away_result_{day+1}')

    #Creo un column transformer, rende più facile normalizzare il dataframe
    ct = make_column_transformer(
    (MinMaxScaler(), numerical_columns), #normalizza queste colonne con minmax
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns) ,# le colonne da one hot
        #encodare, tutte le altre le ignora grazie al comando handle_unknown
    sparse_threshold=0  
    )

    ct.fit(X_train_df)

    # trasformo train e test
    X_train_norm = ct.transform(X_train_df)
    X_valid_norm  = ct.transform(X_valid_df)
    X_test_norm = ct.transform(X_test_df)

    # Crea un OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    # Addestra l'OrdinalEncoder su Train_labels e applica la trasformazione
    Train_labels_encoded = ordinal_encoder.fit_transform(np.array(Train_labels).reshape(-1, 1))

    Train_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Train_labels).reshape(-1, 1)))
    Valid_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Valid_labels).reshape(-1, 1)))
    Test_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Test_labels).reshape(-1, 1)))  

    return  (X_train_norm, X_valid_norm, X_test_norm, Train_labels_encoded, Valid_labels_encoded, Test_labels_encoded, 
               X_train_df, X_valid_df, X_test_df, Train_labels, Valid_labels, Test_labels)



def preprocess_columns_with_odds(dataframe,numero_colonne,giorni_cumulativi, oversample=False):
    all_columns = ['giornata', 'stagione','hometeam', 'awayteam','home_total_points','home_result', 'away_total_points', f'home_last_{giorni_cumulativi}_days_ft_goals',
               f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ht_goals',f'away_last_{giorni_cumulativi}_days_ht_goals', f'away_last_{giorni_cumulativi}_days_shots',
       f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',
       f'home_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots_target',
       f'away_last_{giorni_cumulativi}_days_shots_target',f'home_last_{giorni_cumulativi}_days_fouls_done',f'away_last_{giorni_cumulativi}_days_fouls_done',f'home_last_{giorni_cumulativi}_days_corners_obtained',
       f'away_last_{giorni_cumulativi}_days_corners_obtained',f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows',
       f'home_last_{giorni_cumulativi}_days_reds', f'away_last_{giorni_cumulativi}_days_reds', 'home_win_odds', 'draw_odds', 'away_win_odds']

    less_columns = ['stagione','hometeam', 'awayteam','home_total_points','home_result', 'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',
       f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'home_last_{giorni_cumulativi}_days_shots',
       f'away_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows',
       'home_win_odds', 'draw_odds', 'away_win_odds']

    few_columns = ['hometeam', 'awayteam','home_total_points','home_result', 'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',
               f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',
               f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'home_last_{giorni_cumulativi}_days_shots',
               f'away_last_{giorni_cumulativi}_days_shots','home_win_odds', 'draw_odds', 'away_win_odds']


    if numero_colonne == 'all':
       colonne = all_columns
       print('utilizzando tutte le features')
    elif numero_colonne == 'less':
       print('utilizzando meno features')
       colonne = less_columns
    else:
       print('utilizzando poche features')
       colonne=few_columns


    for day in range(giorni_cumulativi):
       colonne.append(f'home_result_{day+1}')
       colonne.append(f'away_result_{day+1}')

    df_stats_serie_A = dataframe[colonne]   

    #creo il train valid test
    X_train_df, X_test_df, Train_labels, Test_labels = train_test_split(df_stats_serie_A.drop(f'home_result', axis=1), df_stats_serie_A[f'home_result'],
                                                                     test_size=0.05, random_state=42)

    X_train_df, X_valid_df, Train_labels, Valid_labels = train_test_split(X_train_df, Train_labels, test_size=0.11, random_state=42)

    Train_odds_df = X_train_df[['home_win_odds','draw_odds','away_win_odds']]
    Valid_odds_df = X_valid_df[['home_win_odds','draw_odds','away_win_odds']]
    Test_odds_df =  X_test_df[['home_win_odds','draw_odds','away_win_odds']]

    X_train_df.drop(columns=['home_win_odds','draw_odds','away_win_odds'],inplace=True)
    X_valid_df.drop(columns=['home_win_odds','draw_odds','away_win_odds'],inplace=True)
    X_test_df.drop(columns=['home_win_odds','draw_odds','away_win_odds'],inplace=True)


    if oversample: #questo permette di fare l'oversampling di cui si parlava prima
      ros = RandomOverSampler()
      X_train_df, Train_labels = ros.fit_resample(X_train_df, Train_labels)

    #print(f'lunghezza dataframe train {len(X_train_df)}, \n lunghezza dataframe valid {len(X_valid_df)}, \n lunghezza dataframe test {len(X_test_df)}')

    all_numerical_columns= [f'home_total_points',f'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ht_goals',
       f'away_last_{giorni_cumulativi}_days_ht_goals', f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',
       f'home_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots_target',f'away_last_{giorni_cumulativi}_days_shots',f'away_last_{giorni_cumulativi}_days_shots_target',
       f'home_last_{giorni_cumulativi}_days_fouls_done',f'away_last_{giorni_cumulativi}_days_fouls_done',f'home_last_{giorni_cumulativi}_days_corners_obtained',
       f'away_last_{giorni_cumulativi}_days_corners_obtained',f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows',f'home_last_{giorni_cumulativi}_days_reds', 
       f'away_last_{giorni_cumulativi}_days_reds']

    less_numerical_columns = [f'home_total_points',f'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',f'away_last_{giorni_cumulativi}_days_ft_goals',
                           f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',
                           f'away_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots',
                           f'home_last_{giorni_cumulativi}_days_yellows',f'away_last_{giorni_cumulativi}_days_yellows']

    few_numerical_columns = [f'home_total_points',f'away_total_points',f'home_last_{giorni_cumulativi}_days_ft_goals',f'away_last_{giorni_cumulativi}_days_ft_goals',f'home_last_{giorni_cumulativi}_days_ft_goals_conceded',
                            f'away_last_{giorni_cumulativi}_days_ft_goals_conceded',f'away_last_{giorni_cumulativi}_days_shots',f'home_last_{giorni_cumulativi}_days_shots']


    all_categorical_columns = ['giornata', 'stagione',f'hometeam', f'awayteam']
    less_categorical_columns = ['stagione',f'hometeam', f'awayteam']
    few_categorical_columns = [f'hometeam', f'awayteam']

    if numero_colonne == 'all':
        categorical_columns = all_categorical_columns
        numerical_columns = all_numerical_columns
    elif numero_colonne == 'less':
        categorical_columns = less_categorical_columns
        numerical_columns = less_numerical_columns
    else:
        categorical_columns=few_categorical_columns
        numerical_columns = few_numerical_columns


    for day in range(giorni_cumulativi):
        categorical_columns.append(f'home_result_{day+1}')
        categorical_columns.append(f'away_result_{day+1}')

    #Creo un column transformer, rende più facile normalizzare il dataframe
    ct = make_column_transformer(
    (MinMaxScaler(), numerical_columns), #normalizza queste colonne con minmax
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns) ,# le colonne da one hot
        #encodare, tutte le altre le ignora grazie al comando handle_unknown
    sparse_threshold=0  
    )

    ct.fit(X_train_df)

    # trasformo train e test
    X_train_norm = ct.transform(X_train_df)
    X_valid_norm  = ct.transform(X_valid_df)
    X_test_norm = ct.transform(X_test_df)

    odds_transform = make_column_transformer(
    (MinMaxScaler(), ['home_win_odds','draw_odds','away_win_odds']), #normalizza queste colonne con minmax
    sparse_threshold=0  
    )
    odds_transform.fit(Train_odds_df)
    
    Train_odds_df = odds_transform.transform(Train_odds_df)
    Valid_odds_df = odds_transform.transform(Valid_odds_df)
    Test_odds_df = odds_transform.transform(Test_odds_df)

    # Crea un OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    # Addestra l'OrdinalEncoder su Train_labels e applica la trasformazione
    Train_labels_encoded = ordinal_encoder.fit_transform(np.array(Train_labels).reshape(-1, 1))

    Train_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Train_labels).reshape(-1, 1)))
    Valid_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Valid_labels).reshape(-1, 1)))
    Test_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Test_labels).reshape(-1, 1)))  

    return  (X_train_norm, X_valid_norm, X_test_norm, Train_labels_encoded, Valid_labels_encoded, Test_labels_encoded, 
               X_train_df, X_valid_df, X_test_df, Train_labels, Valid_labels, Test_labels,Train_odds_df,Valid_odds_df,Test_odds_df)