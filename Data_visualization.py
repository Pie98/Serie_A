import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

def visualize_home_win_season(dataframe, season = 'Random', team='Random'):
    df_win_loss = dataframe[['stagione','hometeam','awayteam','home_result']].reset_index(drop=True)

    if team == 'Random':
        num_plots= 2
        fig, axs = plt.subplots(num_plots, num_plots, figsize=(24, 14))
    else:
        num_plots=1
        

    if season == 'Random':
        stagione = random.choice(df_win_loss['stagione'].unique())
    else:
        stagione = season   
         
    df_win_loss = df_win_loss[df_win_loss['stagione']==stagione]

    for i in range(num_plots):
        for j in range(num_plots):
            if team == 'Random':
                squadra = random.choice(df_win_loss['hometeam'].unique())
            else:
                squadra=team    

            df_squadra_temp = df_win_loss[(df_win_loss['hometeam'] == squadra) | (df_win_loss['awayteam'] == squadra)]
            vittorie={'home':0,'away':0} 
            pareggi={'home':0,'away':0} 
            sconfitte={'home':0,'away':0} 
            for row in df_squadra_temp.itertuples():
                _,_, home_team, away_team, home_result = row
                if home_team == squadra:
                    if home_result == 'W':
                        vittorie['home'] = vittorie['home']+1
                    elif  home_result == 'D':
                        pareggi['home'] = pareggi['home']+1
                    elif home_result == 'L':
                        sconfitte['home'] = sconfitte['home']+1
                else:
                    if home_result == 'L':
                        vittorie['away'] = vittorie['away']+1
                    elif  home_result == 'D':
                        pareggi['away'] = pareggi['away']+1
                    elif home_result == 'W':
                        sconfitte['away'] = sconfitte['away']+1

            statistiche_casa=pd.DataFrame({'wins':vittorie,'draws':pareggi,'losses':sconfitte}).transpose()

            # Aggiungi il subplot corrente
            if team =='Random':
                ax = axs[i, j]
                # Disegna il grafico nel subplot corrente
                statistiche_casa.plot(kind="bar", figsize=(10, 7), ax=ax,  color=['darkblue', 'lightblue'])
                ax.set_title(f'Stagione: {stagione}, Squadra: {squadra}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                # Aumenta la distanza verticale tra i subplot
                fig.subplots_adjust(hspace=0.3)    
            else:    
                statistiche_casa.plot(kind="bar", figsize=(10, 7), color=['darkblue', 'lightblue'])
                plt.title(f'Stagione: {stagione}, Squadra: {squadra}')

    plt.show()
    return 0


def visualize_gol_wins_last_5(dataframe):

  fig, axs = plt.subplots(2, 2, figsize=(24, 14))

  for i in range(2):
    for j in range(2):
      # calcoliamo i percentili di tiri per ogni partita
      stagione = random.choice(dataframe['stagione'].unique())
      #prendo le righe dalla 50 in poi perché nelle prime 5 giornate non ci sono i 5 giorni precedenti valorizzati
      df_anno = dataframe[dataframe['stagione']==stagione][['home_last_5_days_shots_target','away_last_5_days_shots_target','home_result']].iloc[49:]
      media_tiri_ultime_5 = list(df_anno['home_last_5_days_shots_target'])+ list(df_anno['away_last_5_days_shots_target'])

      #percentile_tiri = np.percentile(media_tiri_ultime_5)
      percentile_tiri = np.round(np.median(media_tiri_ultime_5),0)

      #creo due nuove colonne con i percentili 
      conditions_1 = [
      (df_anno['home_last_5_days_shots_target'] <= percentile_tiri),  
      #( (percentile_tiri[0] <= df_anno['home_last_5_days_shots_target']) & (df_anno['home_last_5_days_shots_target'] < percentile_tiri[1] )), 
      #( (percentile_tiri[1] <= df_anno['home_last_5_days_shots_target']) & (df_anno['home_last_5_days_shots_target'] <= percentile_tiri[2] )), 
      (df_anno['home_last_5_days_shots_target'] > percentile_tiri)
      ]

      conditions_2 = [
      (df_anno['away_last_5_days_shots_target'] <= percentile_tiri),  
      #( (percentile_tiri[0] <= df_anno['away_last_5_days_shots_target']) & (df_anno['away_last_5_days_shots_target'] < percentile_tiri[1] )), 
      #( (percentile_tiri[1] <= df_anno['away_last_5_days_shots_target']) & (df_anno['away_last_5_days_shots_target'] <= percentile_tiri[2] )), 
      (df_anno['away_last_5_days_shots_target'] > percentile_tiri)
      ]

      # Valori corrispondenti alle condizioni
      values = [f'<={percentile_tiri}',
                #f'fra {percentile_tiri[1]} e {percentile_tiri[0]}', f'fra {percentile_tiri[1]} e {percentile_tiri[2]}',
                  f'> {percentile_tiri}']
      # Creazione della nuova colonna 
      df_anno['home_last_5_days_shot_count'] = np.select(conditions_1, values)
      df_anno['away_last_5_days_shot_count'] = np.select(conditions_2, values)


      df_anno_home = df_anno[['home_last_5_days_shot_count','home_result']].groupby('home_last_5_days_shot_count')['home_result'].value_counts().unstack()
      df_anno_away = df_anno[['away_last_5_days_shot_count','home_result']].groupby('away_last_5_days_shot_count')['home_result'].value_counts().unstack()

      df_tiri_vittorie = df_anno_home[['L','D','W']]
      df_tiri_vittorie['W'] = df_anno_home['W'] + df_anno_away['L']
      df_tiri_vittorie['L'] = df_anno_home['L'] + df_anno_away['W']
      df_tiri_vittorie['D'] = df_anno_home['D'] + df_anno_away['D']

      ax = axs[i, j]
      # Disegna il grafico nel subplot corrente
      ax = df_tiri_vittorie.plot(kind="bar", figsize=(10, 7), color=['red','lightblue','green'],ax=ax, title=f'stagione={stagione}')
      ax.legend(bbox_to_anchor=(1.0, 1.0))
      fig.subplots_adjust(hspace=0.4,wspace=0.4)   
      # Impostare i label dell'asse x non ruotati
      ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
      # Assegnare un nome all'asse x
      ax.set_xlabel("shots last 5 matches")
      

  plt.show()


def visualize_wins__yellows_last_5(dataframe):

  fig, axs = plt.subplots(2, 2, figsize=(24, 14))

  for i in range(2):
    for j in range(2):
      # calcoliamo i percentili di tiri per ogni partita
      stagione = random.choice(dataframe['stagione'].unique())
      #prendo le righe dalla 50 in poi perché nelle prime 5 giornate non ci sono i 5 giorni precedenti valorizzati
      df_anno = dataframe[dataframe['stagione']==stagione][['home_last_5_days_yellows','away_last_5_days_yellows','home_result']].iloc[49:]
      media_tiri_ultime_5 = list(df_anno['home_last_5_days_yellows'])+ list(df_anno['away_last_5_days_yellows'])

      #percentile_tiri = np.percentile(media_tiri_ultime_5)
      percentile_tiri = np.round(np.median(media_tiri_ultime_5),0)

      #creo due nuove colonne con i percentili 
      conditions_1 = [
      (df_anno['home_last_5_days_yellows'] <= percentile_tiri),  
      #( (percentile_tiri[0] <= df_anno['home_last_5_days_yellows']) & (df_anno['home_last_5_days_yellows'] < percentile_tiri[1] )), 
      #( (percentile_tiri[1] <= df_anno['home_last_5_days_yellows']) & (df_anno['home_last_5_days_yellows'] <= percentile_tiri[2] )), 
      (df_anno['home_last_5_days_yellows'] > percentile_tiri)
      ]

      conditions_2 = [
      (df_anno['away_last_5_days_yellows'] <= percentile_tiri),  
      #( (percentile_tiri[0] <= df_anno['away_last_5_days_yellows']) & (df_anno['away_last_5_days_yellows'] < percentile_tiri[1] )), 
      #( (percentile_tiri[1] <= df_anno['away_last_5_days_yellows']) & (df_anno['away_last_5_days_yellows'] <= percentile_tiri[2] )), 
      (df_anno['away_last_5_days_yellows'] > percentile_tiri)
      ]

      # Valori corrispondenti alle condizioni
      values = [f'<={percentile_tiri}',
                #f'fra {percentile_tiri[1]} e {percentile_tiri[0]}', f'fra {percentile_tiri[1]} e {percentile_tiri[2]}',
                  f'> {percentile_tiri}']
      # Creazione della nuova colonna 
      df_anno['home_last_5_days_yellow_count'] = np.select(conditions_1, values)
      df_anno['away_last_5_days_yellows'] = np.select(conditions_2, values)


      df_anno_home = df_anno[['home_last_5_days_yellow_count','home_result']].groupby('home_last_5_days_yellow_count')['home_result'].value_counts().unstack()
      df_anno_away = df_anno[['away_last_5_days_yellows','home_result']].groupby('away_last_5_days_yellows')['home_result'].value_counts().unstack()

      df_tiri_vittorie = df_anno_home[['L','D','W']]
      df_tiri_vittorie['W'] = df_anno_home['W'] + df_anno_away['L']
      df_tiri_vittorie['L'] = df_anno_home['L'] + df_anno_away['W']
      df_tiri_vittorie['D'] = df_anno_home['D'] + df_anno_away['D']

      ax = axs[i, j]
      # Disegna il grafico nel subplot corrente
      ax = df_tiri_vittorie.plot(kind="bar", figsize=(10, 7), color=['red','lightblue','green'],ax=ax, title=f'stagione={stagione}')
      ax.legend(bbox_to_anchor=(1.0, 1.0))
      fig.subplots_adjust(hspace=0.4,wspace=0.4)   
      # Impostare i label dell'asse x non ruotati
      ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
      # Assegnare un nome all'asse x
      ax.set_xlabel("shots last 5 matches")
      

  plt.show()