import matplotlib.pyplot as plt
import random
import pandas as pd

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