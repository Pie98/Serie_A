import pandas as pd 
import os
import pandasql as ps

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
                    `Div`,
                    CEIL( (ROW_NUMBER() OVER (PARTITION BY stagione ORDER BY Date))/10) AS giornata,
                    stagione,
                    Date,
                    HomeTeam,
                    AwayTeam,
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
    '''

    df_giornate = ps.sqldf(query_giornate, locals()).fillna(0)

    return df_giornate




preprocess_match_days(r"C:\Users\Hp\Documents\Serie_A\csv_serie_a")
