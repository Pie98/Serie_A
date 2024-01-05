# Serie A Games Predictions

## Overview

This project is a machine learning application (mainly developed with Tensorflow) designed to predict the outcomes of Serie A football matches. By leveraging historical match data as input, the model aims to provide predictions for upcoming Serie A games.

## Table of Contents

- [Project Structure](#Project-Structure)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Project Structure

The repository is organized as follows:

- **csv_predictions:** This folder contains CSV files that will be used as a basis for creating new predictions.

- **csv_serie_a:** Here, you can find CSV files (divided by season) used for training the models.

- **cumulative_stats:** This directory represents an alternative approach to the problem, where the goal is to predict match outcomes after creating a dataset with cumulative results from the last X matches for each team (e.g., Milan scored 10 goals in the last 5 matches).

- **model_experiments:** This folder stores the models created during experimentation.

- **notebooks:** In this directory, you'll find supporting Jupyter notebooks useful for testing.

- **Refreshing_odds:** This folder contains scripts for updating match odds and generating predictions for upcoming matches.

- **results:** Here, certain results obtained during the model training phase are saved.

- **transformers:** The necessary column transformers for evaluating new matches are saved in this directory.


