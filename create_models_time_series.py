import pandas as pd 
import numpy as np
import pandasql as ps
import os
import re
import random 
import numpy as np
import warnings
import tensorflow as tf 
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import make_column_transformer
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def create_time_series_model_dense(Train_teams_shape, feature_input_shape, first_dropout, second_dropout,concat_dropout_1,concat_dropout_2 ):
    #Modello per i teams 
    inputs = layers.Input(shape=(Train_teams_shape,))
    x = layers.Dense(16, activation = 'relu')(inputs)
    outputs = layers.Dense(8)(x)
    model_teams = tf.keras.Model(inputs,outputs, name = 'model_1_teams')

    # modello ft_goals
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_ft_goals = tf.keras.Model(inputs, outputs, name='model_1_goals')

    # modello ft_goals_conceded
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_ft_goals_conceded = tf.keras.Model(inputs, outputs, name='ft_goals_conceded')

    # modello shots
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_shots = tf.keras.Model(inputs, outputs, name='model_1_shots')

    # modello shots_target
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_shots_target = tf.keras.Model(inputs, outputs, name='model_1_shots_target')

    #Unisco i modelli dei tiri 
    model_1_shots_concat_layer = layers.Concatenate(name="shots_concat")([model_shots.output, model_shots_target.output])
    output_layer_shots_concat = layers.Dense(16, activation='relu')(model_1_shots_concat_layer)

    #creo il modello  finale
    model_1_shots_concat =tf.keras.Model(
        inputs=[[ model_shots.input, model_shots_target.input]],
        outputs=output_layer_shots_concat,
        name='model_1_shots_concat'
    )

    # modello fouls_done
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_fouls_done = tf.keras.Model(inputs, outputs, name='model_1_fouls_done')

    # modello corners_obtained
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_corners_obtained = tf.keras.Model(inputs, outputs, name='model_1_corners_obtained')

    # modello yellows
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_yellows = tf.keras.Model(inputs, outputs, name='model_1_corners_yellows')

    # modello reds
    inputs = layers.Input(shape=(feature_input_shape,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(first_dropout)(x)  # Aggiunto il layer di dropout per ridurre overfitting
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dropout(second_dropout)(x) 
    model_reds = tf.keras.Model(inputs, outputs, name='model_1_corners_reds')

    #Unisco i modelli 
    model_1_concat_layer = layers.Concatenate(name="feature_concat")([ model_ft_goals.output, model_ft_goals_conceded.output, 
                                                            model_1_shots_concat.output, model_fouls_done.output, 
                                                            model_corners_obtained.output, model_yellows.output, model_reds.output])
    x = layers.Dense(64, activation='relu')(model_1_concat_layer)
    x = layers.Dropout(concat_dropout_1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(concat_dropout_2)(x)
    output_layer = layers.Dense(3, activation = 'softmax')(x)

    #creo il modello  finale
    model_1_final =tf.keras.Model(
        inputs=[[ model_ft_goals.input, model_ft_goals_conceded.input, model_shots.input, model_shots_target.input, model_fouls_done.input, 
                    model_corners_obtained.input, model_yellows.input, model_reds.input]],
        outputs=output_layer,
        name='model_1_dense_concat'
    )


    #compilo il modelllo 
    model_1_final.compile(
        loss = 'categorical_crossentropy',
        metrics = 'accuracy',
        optimizer = tf.keras.optimizers.Adam()
        )

    return model_1_concat_layer