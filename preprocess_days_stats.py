import pandas as pd 
import numpy as np
import pandasql as ps
import os
    

def preprocess_cumulative_stats(dataframe = [], directory = [], giorni_cumulativi = 5):

    if (len(directory) != 0):
        df_giornate = pd.read_csv(directory, parse_dates=['date'], index_col='index')
    elif (len(dataframe) !=0):   
        df_giornate = dataframe
    else:
        print('Nessun input valido ricevuto')
        return 0   
    
    # Gestione dei valori nulli
    # Rimpiazzo i valori nulli delle odds con 0 e i valori nulli degli altri campi con la loro media
    colonne_nulle = ['away_shots','home_shots','away_shots_targ','home_shots_targ','away_corners','home_corners','away_fouls','home_fouls','ht_away_goals','ht_home_goals','home_yellow','away_yellow','home_red','away_red']
    odds = ['draw_odds','home_win_odds','away_win_odds']
    
    for colonna in colonne_nulle:
        media_colonna = int(df_giornate[colonna].mean())
        df_giornate[colonna].fillna(media_colonna, inplace=True)    
    		
    for odd in odds:
        df_giornate[odd].fillna(0, inplace=True)   
    
    df_giornate['ht_results'].fillna('###', inplace=True)     
    
    
    

    # ## Per ogni squadra creo un dataframe specifico 
    # ### Controllo che le squadre siano coerenti sia in casa che trasferta
    print(f"\nle squadre sono uguali:\n {np.sort(df_giornate['hometeam'].unique()) == np.sort(df_giornate['awayteam'].unique())}")
    
    squadre = np.sort(df_giornate['hometeam'].unique())
    
    # Creo un dataframe per ogni squadra e risultati delle ultime giornate
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
    
    #
    Statistiche_squadre_dict = {}
    
    for squadra in squadre:
        # per ogni squadra creo un dizionario dove ad ogni squadra associo un dataframe 
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
    
        # Valori corrispondenti alle condizioni
        values = ['W', 'L', 'D']
        punti = [3,0,1]
        # Creazione della nuova colonna 'result' e 'points
        Statistiche_squadre_dict[squadra]['points'] = np.select(conditions, punti)
        Statistiche_squadre_dict[squadra]['result'] = np.select(conditions, values)
    
        #Creo colonne con i risultati shiftati in modo che possa avere i risultati comulativi
        for i in range(giorni_cumulativi):
             partita_passata = f'result_{i+1}'
             Statistiche_squadre_dict[squadra][partita_passata] = Statistiche_squadre_dict[squadra].groupby('stagione')['result'].shift(i+1).fillna('###')
    
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
    
    # Creo delle statistiche cumulative per ogni squadra degli ultimi N giorni
    colonne_numeriche=['ft_goals', 'ht_goals', 'shots', 'shots_target','ft_goals_conceded', 'fouls_done', 'corners_obtained', 'yellows', 'reds']
    
    for squadra in squadre:
        temp_df = Statistiche_squadre_dict[squadra]
        query_cumul = f''' SELECT 
                            SUM(ft_goals) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_ft_goals,
                            SUM(ht_goals) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_ht_goals,
                            SUM(shots) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_shots,
                            SUM(shots_target) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_shots_target,
                            SUM(ft_goals_conceded) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_ft_goals_conceded,
                            SUM(fouls_done) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_fouls_done,
                            SUM(corners_obtained) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_corners_obtained,
                            SUM(yellows) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_yellows,
                            SUM(reds) OVER (PARTITION BY stagione ORDER BY date ROWS BETWEEN {giorni_cumulativi} PRECEDING AND 1 PRECEDING ) AS last_{giorni_cumulativi}_days_reds,
                            SUM(points_shift) OVER (PARTITION BY stagione ORDER BY date) AS total_points
                        FROM (
                            SELECT
                                *,
                                LAG(points, 1) OVER (PARTITION BY stagione ORDER BY date) AS points_shift
                            FROM temp_df
                            ) AS subquery'''
        
        #creo il dataframe con le colonne comulative di ogni squadra da aggiungere    
        df_new_column = ps.sqldf(query_cumul, locals()).fillna(0)
        
        Statistiche_squadre_dict[squadra]['total_points'] = df_new_column['total_points']
        #aggiungo le nuove colonne ai dataframe delle squadre
        for campo in colonne_numeriche:  
            campo_cumulativo = 'last_'+str(giorni_cumulativi)+'_days_'+campo
            Statistiche_squadre_dict[squadra][campo_cumulativo] = df_new_column[campo_cumulativo]
            
    # Unisco tutti i dataframe in un unico solo 
    df_squadre_cumul = Statistiche_squadre_dict[squadre[0]]
    for squadra in squadre[1:]:
        df_squadre_cumul = pd.concat([df_squadre_cumul, Statistiche_squadre_dict[squadra]], ignore_index=True)
    
    df_squadre_cumul = df_squadre_cumul.sort_values(by='date')
    df_squadre_cumul.head(10)

    # Unisco le statistiche cumulative al df di partenza
    # Inizializzazione delle stringhe cumulative
    risultati_passati_casa = ''
    risultati_passati_trasferta = ''
    
    # Creazione delle query per risultati cumulativi
    for i in range(giorni_cumulativi):
        risultati_passati_casa += f' home_teams.result_{i+1} AS home_result_{i+1}, '
        risultati_passati_trasferta += f' away_teams.result_{i+1} AS away_result_{i+1}, '
    
    # creo la query per unire le colonne comulative ad ogni giornata
    query_merge = f'''
                    SELECT 
                        giornate.div,
    					giornate.giornata,
    					giornate.stagione,
    					giornate.date,
    					giornate.hometeam,
    					giornate.awayteam,
                        home_teams.total_points AS home_total_points,
    					home_teams.result AS home_result,
    					{risultati_passati_casa}
                        away_teams.total_points AS away_total_points,
    					away_teams.result AS away_result,
                        {risultati_passati_trasferta}
    					giornate.ft_home_goals,
    					home_teams.last_{giorni_cumulativi}_days_ft_goals  AS home_last_{giorni_cumulativi}_days_ft_goals,
    					giornate.ft_away_goals,
    					away_teams.last_{giorni_cumulativi}_days_ft_goals  AS away_last_{giorni_cumulativi}_days_ft_goals,
    					giornate.ft_result,
    					giornate.ht_home_goals,
    					home_teams.last_{giorni_cumulativi}_days_ht_goals  AS home_last_{giorni_cumulativi}_days_ht_goals,
    					giornate.ht_away_goals,
    					away_teams.last_{giorni_cumulativi}_days_ht_goals  AS away_last_{giorni_cumulativi}_days_ht_goals,
                        home_teams.last_{giorni_cumulativi}_days_ft_goals_conceded  AS home_last_{giorni_cumulativi}_days_ft_goals_conceded,
                        away_teams.last_{giorni_cumulativi}_days_ft_goals_conceded  AS away_last_{giorni_cumulativi}_days_ft_goals_conceded,
    					giornate.ht_results,
    					giornate.home_shots,
    					home_teams.last_{giorni_cumulativi}_days_shots  AS home_last_{giorni_cumulativi}_days_shots,
    					giornate.away_shots,
    					away_teams.last_{giorni_cumulativi}_days_shots  AS away_last_{giorni_cumulativi}_days_shots,
    					giornate.home_shots_targ,
    					home_teams.last_{giorni_cumulativi}_days_shots_target  AS home_last_{giorni_cumulativi}_days_shots_target,
    					giornate.away_shots_targ,
    					away_teams.last_{giorni_cumulativi}_days_shots_target  AS away_last_{giorni_cumulativi}_days_shots_target,
    					giornate.home_fouls,
    					home_teams.last_{giorni_cumulativi}_days_fouls_done  AS home_last_{giorni_cumulativi}_days_fouls_done,
    					giornate.away_fouls,
    					away_teams.last_{giorni_cumulativi}_days_fouls_done  AS away_last_{giorni_cumulativi}_days_fouls_done,
    					giornate.home_corners,
    					home_teams.last_{giorni_cumulativi}_days_corners_obtained  AS home_last_{giorni_cumulativi}_days_corners_obtained,
    					giornate.away_corners,
    					away_teams.last_{giorni_cumulativi}_days_corners_obtained  AS away_las_t{giorni_cumulativi}_days_corners_obtained,
    					giornate.home_yellow,
    					home_teams.last_{giorni_cumulativi}_days_yellows  AS home_last_{giorni_cumulativi}_days_yellows,
    					giornate.away_yellow,
    					away_teams.last_{giorni_cumulativi}_days_yellows  AS away_last_{giorni_cumulativi}_days_yellows,
    					giornate.home_red,
    					home_teams.last_{giorni_cumulativi}_days_reds  AS home_last_{giorni_cumulativi}_days_reds,
    					giornate.away_red,
    					away_teams.last_{giorni_cumulativi}_days_reds  AS away_last_{giorni_cumulativi}_days_reds,
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
    return df_Serie_A, Statistiche_squadre_dict
    
def preprocess_match_days(directory_path):
    concatenated_df = pd.DataFrame({})
    #leggo il path di ogni csv e lo converto in dataframe
    for file_name in os.listdir(fr"{directory_path}"):
        if file_name.endswith('.csv'):
            print(f'Reading file: {file_name}')
            file_path = os.path.join(directory_path, file_name)

        temp_df = pd.read_csv(file_path ,parse_dates=['Date'], dayfirst=True )
        temp_df['Date'] = temp_df['Date'].dt.strftime('%d/%m/%Y')
        
        # Concatenare i DataFrame
        concatenated_df = pd.concat([concatenated_df,temp_df], ignore_index=True)

    #prendo solo le colonne importanti al fine della statistica e ordino per data il df
    concatenated_df_important = concatenated_df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG','FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                                'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC',
                                                 'AC', 'HY', 'AY', 'HR', 'AR','B365H','B365D','B365A']]
    
    concatenated_df_important = concatenated_df_important.dropna(subset=['HomeTeam']) #se manca la quadra di casa la statistica è inutile
    concatenated_df_important['Date'] = pd.to_datetime(concatenated_df_important['Date'])
    df_importanti = concatenated_df_important.sort_values(by='Date', ascending=False)    

    query_giornate = f''' 
                    SELECT 
                    `Div` AS div,
                    CEIL(CAST((ROW_NUMBER() OVER (PARTITION BY stagione ORDER BY Date)) AS float)/10) AS giornata,
                    stagione AS stagione,
                    Date AS date,
                    HomeTeam AS hometeam,
                    AwayTeam as awayteam,
                    FTHG AS ft_home_goals,
                    FTAG AS ft_away_goals,
                    FTR AS ft_result,
                    HTHG ht_home_goals,
                    HTAG ht_away_goals,
                    HTR ht_results,
                    HS home_shots,
                    `AS` AS away_shots,
                    HST AS home_shots_targ,
                    AST AS away_shots_targ,
                    HF AS home_fouls,
                    AF AS away_fouls,
                    HC home_corners,
                    AC away_corners,
                    HY home_yellow,
                    AY away_yellow,
                    HR home_red,
                    AR away_red,
                    B365H home_win_odds,
                    B365D draw_odds,
                    B365A away_win_odds
                    FROM (
                        SELECT 
                            *,
                            CASE
                                WHEN strftime('%m', Date) < '08' THEN strftime('%Y', Date, '-1 year') || '/' || strftime('%Y', Date)
                                WHEN strftime('%m', Date) = '08' AND strftime('%d', Date) < '10' THEN strftime('%Y', Date, '-1 year') || '/' || strftime('%Y', Date)
                                ELSE strftime('%Y', Date) || '/' || strftime('%Y', Date, '+1 year')
                            END AS Stagione,
                            Date    
                        FROM df_importanti PSA )
                    ORDER BY date, hometeam    
    '''

    df_giornate = ps.sqldf(query_giornate, locals())
    df_giornate['date'] = pd.to_datetime(df_giornate['date'])
    df_giornate['giornata'] = df_giornate['giornata'].astype(int)
    print('preprocessing finished!')
    
    return df_giornate   
    
    