# Serie A Games Predictions

## Overview

This project is a machine learning application (mainly developed with Tensorflow) designed to predict the outcomes of Serie A football matches. By leveraging historical match data as input, the model aims to provide predictions for upcoming Serie A games.

## Table of Contents

- [Project Structure](#Project-Structure)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Project Structure

The repository is organized as follows:

### Main folder

- **preprocess_days_stats:** The first preprocessing layer. 
    In preprocess_match_days we select the important statistics, rename the columns, infer the date datatype and create the Season field.

-**preprocess_time serie:** The second preprocessing layer. 
    In preprocess_teams we create a dictionary of dataframe. Each dataframe contains the statistics of each match day, for every serie A team
    In create_time_series_features we create a dataframe with the statisctics of every match day and the previous "giorni_cumulativi" match days.

- **preprocess_days_stats:** The last preprocessing layer.
    In preprocess_features_time_series we create all the features and labels necessary to train our model
    In create_fast_preprocessing_ts we batch and prefetch the datasets to accelerate the computation
    In preprocess_features_time_series_odds we batch and prefetch the datasets to accelerate the computation
    In create_fast_preprocessing_ts_odds we create all the features and labels necessary to train our model using the odds as features

- **Time_series_win_predict:** The MAIN notebook where we create our models and test them.

- **create_models_time_series:** The Main script containing the functions to create our models.

- **New_predicts:** A notebook where we calculate the predictions and new odds.

- **New_predicts_calc:** A script containing the functions to calculate predictions and new odds.

-**exploratory_data_analysis:** A notebook where visualize some of the statistics of our dataframe

-**Data_visualization:** A script containing the functions to visualize some statistics

-**helper_functions:** A script containing some useful functions.

### Subfolders

- **csv_predictions:** This folder contains CSV files that will be used as a basis for creating new predictions.

- **csv_serie_a:** Here, you can find CSV files (divided by season) used for training the models.

- **cumulative_stats:** This directory represents an alternative approach to the problem, where the goal is to predict match outcomes after creating a dataset with cumulative results from the last X matches for each team (e.g., Milan scored 10 goals in the last 5 matches).

- **model_experiments:** This folder stores the models created during experimentation.

- **notebooks:** In this directory you'll find Jupyter notebooks useful for testing.

- **Refreshing_odds:** This folder contains scripts for updating match odds and generating predictions for upcoming matches.

- **results:** Here are saved certain results obtained during the model training phase.

- **transformers:** In this directory we can find the necessary column transformers for evaluating new matches.


