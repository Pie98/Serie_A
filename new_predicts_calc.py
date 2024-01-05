import pandas as pd 
import numpy as np
import pandasql as ps
import os
import re
import joblib
import random 
from datetime import datetime
import numpy as np
import csv
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf 
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import make_column_transformer
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from preprocess_days_stats import preprocess_match_days
from preprocess_time_serie import preprocess_teams, create_time_series_features
from preprocess_time_series_features import preprocess_features_time_series, create_fast_preprocessing_ts, preprocess_features_time_series_odds, create_fast_preprocessing_ts_odds
from helper_functions_tensorflow import CSVLoggerCallback, CSVLoggerCallbackParams
from Refreshing_odds.refresh_odds import refresh_odds


#######################################################
 
#---- preprocess features time series odds preds -----#

#######################################################


def preprocess_features_time_series_odds_preds(df_Serie_A, num_features, today_date):

    all_features = ['ft_goals','ft_goals_conceded','shots','shots_target', 'fouls_done','corners_obtained', 'yellows', 'reds']
    less_features = ['ft_goals','ft_goals_conceded','shots', 'fouls_done','corners_obtained', 'reds']
    few_features = ['ft_goals','ft_goals_conceded','shots', 'reds']

    Train_df = df_Serie_A.iloc[:10]
    Valid_df = df_Serie_A.iloc[:10]
    Test_df = df_Serie_A[df_Serie_A['date']==today_date]

    Train_labels = Train_df[['ft_result']]
    Valid_labels = Valid_df[['ft_result']]
    Test_labels = Test_df[['ft_result']]
    
    Train_odds = Train_df[['home_win_odds','draw_odds','away_win_odds']]
    Valid_odds = Valid_df[['home_win_odds','draw_odds','away_win_odds']]
    Test_odds = Test_df[['home_win_odds','draw_odds','away_win_odds']]

    # preprocess Train dataframe
    Train_teams = Train_df[['stagione','hometeam','awayteam']]

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
    Valid_teams = Valid_df[['stagione','hometeam','awayteam']]


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
    Test_teams = Test_df[['stagione','hometeam','awayteam']]

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
    # load the  transformer
    teams_transf = joblib.load('transformers/teams_transformer.pkl')

    Train_teams_encoded = teams_transf.transform(Train_teams)
    Valid_teams_encoded = teams_transf.transform(Valid_teams)
    Test_teams_encoded = teams_transf.transform(Test_teams)

    #encoding labels
    # load the  transformer
    ordinal_encoder = joblib.load('transformers/ordinal_encoder_transformer.pkl')

    Train_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Train_labels).reshape(-1, 1)))
    Valid_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Valid_labels).reshape(-1, 1)))
    Test_labels_encoded = np.squeeze(ordinal_encoder.transform(np.array(Test_labels).reshape(-1, 1)))  

    #encoding numerical features
    Train_dict_features_norm = Train_dict_features.copy()
    Valid_dict_features_norm = Valid_dict_features.copy()
    Test_dict_features_norm = Test_dict_features.copy()

    for feature in list(Train_dict_features.keys()):
        # load the  transformer
        numerical_transf = joblib.load(f'transformers/numerical_{feature}_transformer.pkl')

        Train_dict_features_norm[feature] = numerical_transf.transform(Train_dict_features_norm[feature])
        Valid_dict_features_norm[feature] = numerical_transf.transform(Valid_dict_features_norm[feature])
        Test_dict_features_norm[feature] = numerical_transf.transform(Test_dict_features_norm[feature])
    
    # Encoding odds
    # load the  transformer
    odds_transf = joblib.load('transformers/odds_transformer.pkl')

    Train_odds_norm = odds_transf.transform(Train_odds)
    Valid_odds_norm = odds_transf.transform(Valid_odds)
    Test_odds_norm = odds_transf.transform(Test_odds)

    return (Train_teams_encoded, Valid_teams_encoded, Test_teams_encoded, Train_labels_encoded, Valid_labels_encoded, Test_labels_encoded, 
            Train_dict_features_norm, Valid_dict_features_norm, Test_dict_features_norm, Train_teams, Valid_teams, Test_teams, Train_labels, Valid_labels, Test_labels, 
            Train_dict_features, Valid_dict_features, Test_dict_features, Train_df, Valid_df, Test_df, 
            Train_odds_norm, Valid_odds_norm, Test_odds_norm)


#################################################

#-------------- new predicts calc ------------- #

#################################################

def new_predictions_calc(today_date, home_teams, away_teams, prima_iterazione):
    #obtaining the new odds
    column_types = {'1': float, 'x': float, '2': float}
    last_odds = pd.read_csv(r'Refreshing_odds/last_odds.csv', dtype=column_types)
    previous_odds = pd.read_csv(r'Refreshing_odds/previous_odds.csv', dtype=column_types)

    #adding the new results to the dataframe
    # read new season matches CSV
    df = pd.read_csv(r"C:\Users\Hp\Serie_A\csv_predictions\stagione_23_24.csv", parse_dates=['Date'], dayfirst=True)
    df = df[df['Date'] != today_date]

    # create a dataframe with 10 rows of zeros
    new_rows = pd.DataFrame(0, index=range(len(last_odds)), columns=df.columns)

    #updating teams columns
    for index, row in last_odds.iterrows():
        win, draw, loss = row
        new_rows.loc[index, ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                              'FTR']] = ['I1', f'{today_date}', '17:30', home_teams[index], away_teams[index], 0, 0, 'D']

    #udpating the odds columns
    # new_rows.loc[0,['B365H', 'B365D', 'B365A']] = [4.75, 3.25, 1.85]
    # new_rows.loc[1,['B365H', 'B365D', 'B365A']] = [2.05, 3.35, 3.6]
    # new_rows.loc[2,['B365H', 'B365D', 'B365A']] = [1.3, 5.5, 9.0]
    # new_rows.loc[3,['B365H', 'B365D', 'B365A']] = [1.55, 4.0, 6.0]
    # new_rows.loc[4,['B365H', 'B365D', 'B365A']] = [1.55, 4.25, 5.75]
    # new_rows.loc[5,['B365H', 'B365D', 'B365A']] = [1.45, 4.5, 6.5]
    # new_rows.loc[6,['B365H', 'B365D', 'B365A']] = [2.25, 3.5, 3.05]
    # new_rows.loc[7,['B365H', 'B365D', 'B365A']] = [2.35, 3.0, 3.3]
    # new_rows.loc[8,['B365H', 'B365D', 'B365A']] = [4.75, 3.7, 1.73]
    # new_rows.loc[9,['B365H', 'B365D', 'B365A']] = [1.3, 5.5, 8.75]
        
    for index, row in last_odds.iterrows():
        win, draw, loss = row
        new_rows.loc[index,['B365H', 'B365D', 'B365A']] = [win,draw,loss]
    
    # create a previous and last odds dataframe
    last_odds_df = pd.DataFrame()
    last_odds_df[['HomeTeam','home_odds', 'draw_odds', 'away_odds']] = new_rows[['HomeTeam','B365H', 'B365D', 'B365A']].copy()
    last_odds_df = last_odds_df.set_index('HomeTeam')
    previous_odds_df = last_odds_df.copy()
    for index, row in previous_odds.iterrows():
        win, draw, loss = row
        previous_odds_df.iloc[index, 0:3] = [win,draw,loss]

    # adding the new rows to the existing dataframe
    new_csv = pd.concat([df, new_rows], ignore_index=True)
    new_csv['Date'] = pd.to_datetime(new_csv['Date'], format='%Y-%m-%d')
    if ((len(new_csv['HomeTeam'].unique())!=20) | (len(new_csv['AwayTeam'].unique())!=20)):
        raise ValueError('Errore nel nome di una squadra')

    # Save the new dataframe 
    new_csv.to_csv(r"C:\Users\Hp\Serie_A\csv_predictions\stagione_23_24.csv")

    print(f" numero di squadre: {len(new_csv['HomeTeam'].unique())}, {len(new_csv['AwayTeam'].unique())} ")

    #preprocessing the features
    df_giornate = preprocess_match_days(r"C:\Users\Hp\Serie_A\csv_predictions")
    num_features = 'less'
    num_giornate = 4
    Statistiche_squadre_dict = preprocess_teams(dataframe = df_giornate)
    df_Serie_A = create_time_series_features(num_features, Statistiche_squadre_dict, df_giornate, num_giornate).dropna()

    (Train_teams_encoded, Valid_teams_encoded, Test_teams_encoded, Train_labels_encoded, Valid_labels_encoded, Test_labels_encoded, 
        Train_dict_features_norm, Valid_dict_features_norm, Test_dict_features_norm, Train_teams, Valid_teams, Test_teams, Train_labels, Valid_labels, 
        Test_labels, Train_dict_features, Valid_dict_features, Test_dict_features, Train_df, Valid_df, Test_df, 
        Train_odds_norm, Valid_odds_norm, Test_odds_norm) = preprocess_features_time_series_odds_preds(df_Serie_A, num_features, today_date)

    feature_input_shape = Test_dict_features_norm[list(Test_dict_features_norm.keys())[0]].shape[1]
    Train_teams_shape = Test_teams_encoded.shape[1]

    Dataset_train_norm, Dataset_valid_norm, Dataset_test_norm = create_fast_preprocessing_ts_odds(Train_teams_encoded, Train_dict_features_norm, Train_labels_encoded,
                                                                                            Valid_teams_encoded, Valid_dict_features_norm,
                                                                        Valid_labels_encoded,Test_teams_encoded, Test_dict_features_norm, Test_labels_encoded, 
                                                                        Train_odds_norm, Valid_odds_norm, Test_odds_norm)
    
    #import the model 
    odds_model = tf.keras.models.load_model(r'c:\Users\Hp\Serie_A\model_experiments\model_odds_time_series')

    ## Visualizziamo un po' di risultati 
    model_odds_new_pred_probs = odds_model.predict((Dataset_test_norm))
    model_odds_new_prob = model_odds_new_pred_probs.max(axis=1)
    model_odds_new_predictions = model_odds_new_pred_probs.argmax(axis=1)
    model_odds_new_compare = pd.DataFrame({
                                    'hometeam': list( Test_df['hometeam'] ),
                                    'awayteam': list( Test_df['awayteam'] ),
                                    'preds': model_odds_new_predictions, 
                                    'best_pred_prob': model_odds_new_prob,
                                    })

    model_odds_new_compare[['model_away_prob','model_draw_prob','model_home_prob']] = model_odds_new_pred_probs
    model_odds_new_compare[['home_win_odds', 'draw_odds', 'away_win_odds']] = Test_df[['home_win_odds', 'draw_odds', 'away_win_odds']].values
    model_odds_new_compare['snai_pred'] = np.argmin(Test_df[['home_win_odds', 'draw_odds', 'away_win_odds']].fillna(0.0).to_numpy(), axis=1)
    model_odds_new_compare['snai_prob'] = np.nanmin(Test_df[['home_win_odds', 'draw_odds', 'away_win_odds']].fillna(0.0).to_numpy(), axis=1)

    # Assegno ai valori encoded dei valori più comprensibili per vittoria pareggio sconfitta
    conditions = [
    (model_odds_new_compare['preds'] == 2),  # Condizione per Home Win
    (model_odds_new_compare['preds'] == 0),  # Condizione per Away Win
    (model_odds_new_compare['preds'] == 1)   # Condizione per draw
    ]
    conditions_snai = [
    (model_odds_new_compare['snai_pred'] == 0),  # Condizione per Home Win
    (model_odds_new_compare['snai_pred'] == 2),  # Condizione per Away Win
    (model_odds_new_compare['snai_pred'] == 1)   # Condizione per Draw
    ]
    # Valori corrispondenti alle condizioni
    values = ['H', 'A', 'D']

    # Creazione della nuova colonna 'result' e 'points
    model_odds_new_compare['preds'] = np.select(conditions, values)
    model_odds_new_compare['snai_pred'] = np.select(conditions_snai, values)

    # creo la colonna money won 
    model_odds_new_compare['pred_odds'] = model_odds_new_compare.apply(lambda row: row['home_win_odds'] if row['preds'] == 'H' else (row['draw_odds'] if row['preds'] == 'D' 
                                                                                                else row['away_win_odds']), axis=1)
    model_odds_new_compare['money_won'] = model_odds_new_compare['best_pred_prob']*model_odds_new_compare['pred_odds']
    # Inserisci la colonna nella nuova posizione
    insert_data = model_odds_new_compare['money_won']
    model_odds_new_compare.drop(columns=['money_won'], inplace=True)
    model_odds_new_compare.insert(4, 'money_won', insert_data)
    model_odds_new_compare['money_won_home'] = model_odds_new_compare['model_home_prob']*model_odds_new_compare['home_win_odds']
    model_odds_new_compare['money_won_draw'] = model_odds_new_compare['model_draw_prob']*model_odds_new_compare['draw_odds']
    model_odds_new_compare['money_won_away'] = model_odds_new_compare['model_away_prob']*model_odds_new_compare['away_win_odds']
    model_odds_new_compare['hometeam1'] = model_odds_new_compare['hometeam']
    model_odds_new_compare['awayteam1'] = model_odds_new_compare['awayteam']
    print(f'\n ultima esecuzione : {datetime.now()}')

    #save the predictions in a csv
    if prima_iterazione == False:
        old_predictions = pd.read_csv('predictions_variations.csv')
        old_predictions = old_predictions[list(model_odds_new_compare['hometeam']) + ['Date']]
        old_predictions.to_csv('predictions_variations.csv', index=False)
    with open('predictions_variations.csv', mode='a', newline='', encoding='utf-8',) as file:
        csv_writer = csv.writer(file)
        print(os.path.getsize('predictions_variations.csv'))
        if os.path.getsize('predictions_variations.csv') == 0:
            csv_writer.writerow(list(model_odds_new_compare['hometeam']) + ['Date'])
        csv_writer.writerow(list(model_odds_new_compare['money_won']) + [str(datetime.now().strftime("%d/%m/%y-%H:%M"))])

    return last_odds, previous_odds, model_odds_new_compare
    

#######################################################
 
#------------ Display money win predicts  ------------#

#######################################################


def display_money_win_predict():
    predict_variations = pd.read_csv(r'predictions_variations.csv', parse_dates=['Date'])

    tempi = [(list(predict_variations['Date'])[i] - list(predict_variations['Date'])[0]).total_seconds() / 60 for i in range(len(predict_variations))]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False, gridspec_kw={'hspace': 0.4})
    plt.subplot(2,1,1)
    for squadra in predict_variations.columns[:-1]:
        plt.plot(tempi, predict_variations[squadra])

    plt.legend(predict_variations.columns[:-1], bbox_to_anchor=(1, 1.15))
    plt.xlabel('time difference (mins)')
    plt.ylabel('money win prob')
    plt.title(f"From {list(predict_variations['Date'])[0]} \n   to {list(predict_variations['Date'])[len(predict_variations)-1]}")

    plt.plot()
    # create histogram
    plt.subplot(2,1,2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    differenza_ultima_penultima = predict_variations[predict_variations.columns[:-1]].diff().iloc[-1]
    colori = ['green' if num > 0 else ('red' if num < 0 else 'blue') for num in differenza_ultima_penultima]
    plt.bar(predict_variations.columns[:-1], predict_variations[predict_variations.columns[:-1]].iloc[-1], color=colori)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.title(f"Last odds money-win probability")
    plt.show()

    return predict_variations


###########################################################
 
#------------ Display last and previous odds  ------------#

###########################################################
    
def display_last_previous_odds(last_odds_df, previous_odds_df, home_teams):
    # create bar_plot
    bar_width = 0.2  # width of each bar
    index_values = range(len(last_odds_df.index))
    offsets = [-bar_width, 0, bar_width]  # Offset of each bar 3

    differenza_df = last_odds_df - previous_odds_df
    palette = [['salmon','lightblue', 'lightgreen'], ['red','blue', 'green'], ['darkred','darkblue', 'darkgreen']]
    for i, col in enumerate(last_odds_df.columns):
        colori = [palette[i][0] if diff < 0 else (palette[i][1] if diff == 0 else palette[i][2]) for diff in differenza_df[col]]
        plt.bar([idx + offsets[i] for idx in index_values], last_odds_df[col], width=bar_width, label=col, color=colori)

    plt.xlabel('Indice')
    plt.ylabel('Quote')
    plt.title('Bar Plot con Barre Raggruppate')
    plt.xticks(last_odds_df.index, home_teams, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show() 

    