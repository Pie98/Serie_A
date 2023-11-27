import matplotlib.pyplot as plt
import random
def visualize_home_win_season(dataframe, season = 'Random', team='Random'):
    df_win_loss = dataframe[['stagione','hometeam','awayteam','home_result']]

    if team == 'Random':
        num_plots= 1
    else:
        num_plots=2
    fig, axs = plt.subplots(num_plots, num_plots, figsize=(14, 14))

    if season == 'Random':
        stagione = random.choice(dataframe['stagione'])
    else:
        stagione = team    
    df_win_loss = df_win_loss[df_win_loss['stagione']==stagione]

    for i in range(num_plots):
        for j in range(num_plots):
            if team == 'Random':
                squadra = random.choice(df_win_loss['hometeam'])
            else:
                squadra=team    

            df_squadra_temp = df_win_loss[(df_win_loss['hometeam'] == squadra) | (df_win_loss['awayteam'] == squadra)]
            risultati_casa={'vittorie':0,'pareggi':0,'sconfitte':0} # creo un vettore che 
            risultati_trasferta = {'vittorie':0,'pareggi':0,'sconfitte':0}
            for row in df_squadra_temp.itertuples():
                _,_, home_team, away_team, home_result = row
                if home_team == squadra:
                    if home_result == 'W':
                        risultati_casa['vittorie'] = risultati_casa['vittorie']+1
                    elif  home_result == 'D':
                        risultati_casa['pareggi'] = risultati_casa['pareggi']+1
                    elif home_result == 'L':
                        risultati_casa['sconfitte'] = risultati_casa['sconfitte']+1
                else:
                    if home_result == 'L':
                        risultati_trasferta['vittorie'] = risultati_casa['vittorie']+1
                    elif  home_result == 'D':
                        risultati_trasferta['pareggi'] = risultati_casa['pareggi']+1
                    elif home_result == 'W':
                        risultati_trasferta['sconfitte'] = risultati_casa['sconfitte']+1

            statistiche_casa=pd.DataFrame({'home':risultati_casa,'away':risultati_trasferta}).transpose()

            # Aggiungi il subplot corrente
            ax = axs[i, j]
            # Disegna il grafico nel subplot corrente
            statistiche_casa.plot(kind="bar", figsize=(10, 7), ax=ax, color=['green', 'yellow', 'red'])
            ax.set_title(f'Stagione: {stagione}, Squadra: {squadra}')
            ax.legend(bbox_to_anchor=(1.0, 1.0))