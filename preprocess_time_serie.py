import pandas as pd 
import numpy as np
import pandasql as ps
import os
import random 
import warnings
import tensorflow as tf 

#############################################################

# ---------------- preprocessing teams ---------------------#

#############################################################


def preprocess_teams(dataframe = [], directory = []):
    '''
    inputs: 
        a dataframe containing the statistics of every match day or a directory where a csv with the statistics is sotred.
    outputs: 
        Statistiche_squadre_dict: a dictionary of dataframe. Each dataframe contains the statistics of each match day, for every serie A team
    '''
    if (len(directory) != 0):
        df_giornate = pd.read_csv(directory, parse_dates=['date'], index_col='index')
    elif (len(dataframe) !=0):   
        df_giornate = dataframe
    else:
        print('Nessun input valido ricevuto')
        return 0   
    
    # managing null values 
    # Replace odds null values with 0 and the other null values with their average
    colonne_nulle = ['away_shots','home_shots','away_shots_targ','home_shots_targ','away_corners','home_corners','away_fouls','home_fouls','ht_away_goals','ht_home_goals','home_yellow','away_yellow','home_red','away_red']
    odds = ['draw_odds','home_win_odds','away_win_odds']
    
    for colonna in colonne_nulle:
        media_colonna = int(df_giornate[colonna].mean())
        df_giornate[colonna].fillna(media_colonna, inplace=True)    
    		
    for odd in odds:
        df_giornate[odd].fillna(0, inplace=True)   
    
    df_giornate['ht_results'].fillna('###', inplace=True)     

    #creating a dataframe for each team
    #print(f"\nle squadre sono uguali:\n {np.sort(df_giornate['hometeam'].unique()) == np.sort(df_giornate['awayteam'].unique())}")
    
    squadre = np.sort(df_giornate['hometeam'].unique())

    squadra_list = []
    index_list = []
    div_list = []
    giornata_list = []
    stagione_list = []
    date_list = []
    ft_goal_list = []
    ft_goal_subiti_list = []
    ht_goal_list = []
    tiri_list = []
    tiri_porta_list = []
    falli_commessi_list = []
    corner_favore_list = []
    gialli_list = []
    rossi_list =[]  
    
    Statistiche_squadre_dict = {}
    
    for squadra in squadre:
        # Creating a dictionary where the keys are the teams and the values the features dictionary
        df_squadra_0 = df_giornate[(df_giornate['hometeam'] == squadra) | (df_giornate['awayteam'] == squadra)]
    
        for row in df_squadra_0.itertuples():
    
            index, div, giornata, stagione, date, hometeam, awayteam, ft_home_goals, ft_away_goals, ft_result, ht_home_goals, ht_away_goals, ht_results, home_shots, away_shots, home_shots_targ, away_shots_targ, home_fouls, away_fouls, home_corners, away_corners, home_yellow, away_yellow, home_red, away_red, home_win_odds, draw_odds, away_win_odds = row
            
            squadra_list.append(squadra)
            index_list.append(index)
            div_list.append(div)
            giornata_list.append(giornata)
            stagione_list.append(stagione)
            date_list.append(date) 
    
            if hometeam == squadra: 
                ft_goal_list.append(ft_home_goals)
                ft_goal_subiti_list.append(ft_away_goals)
                ht_goal_list.append(ht_home_goals)
                tiri_list.append(home_shots)
                tiri_porta_list.append(home_shots_targ)
                falli_commessi_list.append(home_fouls)
                corner_favore_list.append(home_corners)
                gialli_list.append(home_yellow)
                rossi_list.append(home_red) 
    
            else:
                ft_goal_list.append(ft_away_goals)
                ft_goal_subiti_list.append(ft_home_goals)
                ht_goal_list.append(ht_away_goals)
                tiri_list.append(away_shots)
                tiri_porta_list.append(away_shots_targ)
                falli_commessi_list.append(away_fouls)
                corner_favore_list.append(away_corners)
                gialli_list.append(away_yellow)
                rossi_list.append(away_red) 
    
        data_squadra = {
        'squadra': squadra_list,
        'index': index_list,
        'div': div_list,
        'giornata': giornata_list,
        'stagione': stagione_list,
        'date': date_list,
        'ft_goals': ft_goal_list,
        'ft_goals_conceded': ft_goal_subiti_list,
        'ht_goals': ht_goal_list,
        'shots': tiri_list,
        'shots_target': tiri_porta_list,
        'fouls_done': falli_commessi_list,
        'corners_obtained': corner_favore_list,
        'yellows': gialli_list,
        'reds': rossi_list }
    
        Statistiche_squadre_dict[squadra] = pd.DataFrame(data_squadra).sort_values(by='date')
    
        conditions = [
        (Statistiche_squadre_dict[squadra]['ft_goals'] > Statistiche_squadre_dict[squadra]['ft_goals_conceded']),  # Condizione per Home Win
        (Statistiche_squadre_dict[squadra]['ft_goals'] < Statistiche_squadre_dict[squadra]['ft_goals_conceded']),  # Condizione per Away Win
        (Statistiche_squadre_dict[squadra]['ft_goals'] == Statistiche_squadre_dict[squadra]['ft_goals_conceded'])   # Condizione per Draw
        ]
    
        # conditions corrisponding to the values
        values = ['W', 'L', 'D']
        punti = [3,0,1]
        # creating a new column 'result' e 'points
        Statistiche_squadre_dict[squadra]['points'] = np.select(conditions, punti)
        Statistiche_squadre_dict[squadra]['result'] = np.select(conditions, values)

        squadra_list = []
        index_list = []
        div_list = []
        giornata_list = []
        stagione_list = []
        date_list = []
        ft_goal_list = []
        ft_goal_subiti_list = []
        ht_goal_list = []
        tiri_list = []
        tiri_porta_list = []
        falli_commessi_list = []
        corner_favore_list = []
        gialli_list = []
        rossi_list =[] 

    return Statistiche_squadre_dict


##################################################################

# ---------------- create_time_series_features ------------------#

##################################################################


def create_time_series_features(num_features, team_stats_dict, df_giornate, giorni_cumulativi ):
   '''
   inputs: 
        df_giornate: a dataframe with the statistics of every match day
        num_features: The number of features that we want to use to train our model
        teams_stats_dict: a dictionary of dataframe. Each dataframe contains the statistics of each match day, for every serie A team
        giorni_cumulativi: The number of previous match days that we want to use to predict the next match day
    outputs: 
         df_Serie_A: A dataframe with the statisctics of every match day and the previous "giorni_cumulativi" match days.
   '''
   all_features = ['ft_goals','ft_goals_conceded','shots','shots_target', 'fouls_done','corners_obtained', 'yellows', 'reds']
   less_features = ['ft_goals','ft_goals_conceded','shots', 'fouls_done','corners_obtained', 'reds']
   few_features = ['ft_goals','ft_goals_conceded','shots', 'reds']

   if num_features == 'all':
      colonne = all_features
      print('utilizzando tutte le features')
   elif num_features == 'less':
      print('utilizzando meno features')
      colonne = less_features
   else:
      print('utilizzando poche features')
      colonne=few_features

   for squadra in team_stats_dict.keys():
      for feature in colonne:
         # Creating columns with shifted results (to obtain past days results)
         for i in range(giorni_cumulativi):
            feature_passata = f'{feature}_{i+1}'
            team_stats_dict[squadra][feature_passata] = team_stats_dict[squadra].groupby('stagione')[feature].shift(i+1)

   # I concatenate all the dataframes
   squadre = list(team_stats_dict.keys())
   df_squadre_cumul = team_stats_dict[squadre[0]]
   for squadra in squadre[1:]:
      df_squadre_cumul = pd.concat([df_squadre_cumul, team_stats_dict[squadra]], ignore_index=True)
   
   df_squadre_cumul = df_squadre_cumul.sort_values(['date'])

   # initializing the select query
   risultati_passati_casa = ''
   risultati_passati_trasferta = ''
   
   # Creazione queries to select the past days features
   for feature in colonne:
       for i in range(giorni_cumulativi):
           risultati_passati_casa += f' home_teams.{feature}_{i+1} AS home_{feature}_{i+1}, '
           risultati_passati_trasferta += f' away_teams.{feature}_{i+1} AS away_{feature}_{i+1}, '
   
   # creo la query per unire le colonne comulative ad ogni giornata
   query_merge = f'''
                   SELECT 
                       giornate.div,
   					giornate.giornata,
   					giornate.stagione,
   					giornate.date,
   					giornate.hometeam,
   					giornate.awayteam,
   					{risultati_passati_casa}
                  {risultati_passati_trasferta}
   					giornate.ft_result,
   					giornate.home_win_odds,
   					giornate.draw_odds,
   					giornate.away_win_odds
                   FROM 
                       df_giornate giornate 
                   LEFT JOIN 
                       df_squadre_cumul home_teams ON home_teams.squadra = giornate.hometeam AND home_teams.date = giornate.date
                   LEFT JOIN 
                       df_squadre_cumul away_teams ON away_teams.squadra = giornate.awayteam AND away_teams.date = giornate.date    
                   '''
   df_Serie_A = ps.sqldf(query_merge, locals()).sort_values(by='date')
   df_Serie_A['date'] = pd.to_datetime(df_Serie_A['date'])
   print('preprocess finished')
   
   return df_Serie_A
    
          

    