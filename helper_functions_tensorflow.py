import csv
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

## classe per freare un file csv per salvare i risultati dei modelli
class CSVLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, filename, experiment_name, overwrite= False):
        self.filename = filename
        self.experiment_name = experiment_name
        self.fieldnames = ['experiment','datetime', 'epoch', 'loss', 'accuracy','val_loss','val_accuracy']  # Aggiungi altre metriche secondo necessità
        self.first_time = overwrite
        
        if self.first_time:
            write_mode = 'w'
        else:
            write_mode = 'a'
        with open(self.filename, mode=write_mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if self.first_time:
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'experiment': self.experiment_name,
            'datetime': current_time,
            'epoch': epoch,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy'),
            # Aggiungi altre metriche secondo necessità
        }

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)

class CSVLoggerCallbackParams(tf.keras.callbacks.Callback):
    def __init__(self, filename, experiment_name, num_dense_layers, first_dropout, other_dropouts, first_num_neurons,
                                    other_num_neurons, first_activation, other_activations, overwrite= False):
        self.filename = filename
        self.experiment_name = experiment_name
        self.num_dense_layers = num_dense_layers
        self.first_dropout = first_dropout
        self.other_dropouts = other_dropouts
        self.first_num_neurons = first_num_neurons
        self.other_num_neurons = other_num_neurons
        self.first_activation = first_activation
        self.other_activations = other_activations
        self.fieldnames = ['experiment', 'num_dense_layers', 'first_dropout', 'other_dropouts', 'first_num_neurons',
                                    'other_num_neurons', 'first_activation', 'other_activations', 'epoch',
                                    'loss', 'accuracy','val_loss','val_accuracy']  # Aggiungi altre metriche secondo necessità
        self.first_time = overwrite
        
        if self.first_time:
            write_mode = 'w'
        else:
            write_mode = 'a'
        with open(self.filename, mode=write_mode, newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            if self.first_time:
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        row = {
            'experiment': self.experiment_name,
            'epoch': epoch,
            'loss': logs.get('loss'),
            'num_dense_layers': self.num_dense_layers,
            'first_dropout': self.first_dropout,
            'other_dropouts': self.other_dropouts,
            'first_num_neurons': self.first_num_neurons,
            'other_num_neurons': self.other_num_neurons,
            'first_activation': self.first_activation,
            'other_activations': self.other_activations,
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy'),
            # Aggiungi altre metriche secondo necessità
        }

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)

#plotto la loss e accuracy curves separatamente
def plot_loss_curve(history):
  '''
  restituisce curve distinte per loss e accuracy
  '''
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  epochs = range(len(history.history['loss']))

  #plot loss
  plt.figure()
  plt.plot(epochs, loss ,label=['training_loss'])
  plt.plot(epochs, val_loss ,label=['val_loss'])
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  #plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy ,label=['training_accuracy'])
  plt.plot(epochs, val_accuracy ,label=['val_accuracy'])
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  return 0


#Funzione per disegnare una confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 7), threshold=None, text_size=15, savefig=False):
    # Calcola la matrice di confusione e la normalizza
    mc = confusion_matrix(y_true, y_pred)
    mc_norm = mc.astype('float') / mc.sum(axis=1)[:, np.newaxis]

    n_classes = mc.shape[0]

    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.matshow(mc, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes is None:
        classes = np.arange(n_classes)

    ax.set(
        title='Confusion Matrix',
        xlabel='Predicted Labels',
        ylabel='True Label',
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes)
    )
    # Ruota le etichette degli assi x e y di 45 gradi
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes, rotation=45)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    if threshold is None:
        threshold = (mc.max() + mc.min()) / 2

    for i, j in itertools.product(range(mc.shape[0]), range(mc.shape[1])):
        plt.text(j, i, f"{mc[i, j]} \n ({mc_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if mc[i, j] > threshold else "black",
                 size=text_size)
        
      # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")    

    return mc_norm

# funzione che prende in input il nome del modello base e il modello di feature extraction da tunare, i layers che voglio tunare e il learning rate 
# e mi restituisce in output il modello compilato
def compile_categorical_model_tuned(model_base, feature_extr_model, unfrozen_layers, learningrate):
    '''
    inputs:
        * base_model (str): Il nome del modello di base usato per la feature extraction, 
          con tutti i layers non trainable
        * feature_extr_model (tf.model): il modello di feature extraction da fine tunare
        * unfrozen_layers (int): Il numero di layer finali che voglio scongelare 
        * Il learning rate da usare durante la compilazione
    outputs: Un modello compilato
    
    warning: modello creato con tensorflow api per usare mixed precision
    '''
    
    #controllo che il base_model non abbia layer già addestrabili
    count_trainable=0
    for layer in feature_extr_model.get_layer(model_base).layers:
        if layer.trainable:
                    count_trainable=count_trainable+1
    print(f'\n\n Ci sono {count_trainable} layers addestrabili nel base_model del feature extraction model')
    
    model = tf.keras.models.clone_model(feature_extr_model)
    model.set_weights(feature_extr_model.get_weights())
    
    #imposto gli ultimi 'unfrozen layers' come addestrabili
    model.trainable = True
    for layer in model.get_layer('base_model').layers[:-unfrozen_layers]:
        layer.trainable=False
        
    #Controllo che i layer addestrabili del modello di base siano quelli che voglio 
    count_trainable=0
    for layer in model.get_layer(model_base).layers:
        if layer.trainable:
                    count_trainable=count_trainable+1
    print(f'Ci sono {count_trainable} layers addestrabili nel base_model del fine tuned base_model \n\n')     
    
    #compilo il modello
    model.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learningrate), # 10x lower learning rate than the default
                        metrics=["accuracy"])
    
    return model


# Creo la funzione di valutazione
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  '''
  Calculates accuracy, precision, recall and f1 score of a binary classification model.

  Args:
    y_true = true labels in the form of a 1D array
    y_pred = predicted labels in the form of a 1D array

  Output:
    Returns a dictionary of accuracy, precision, recall, f1-score.
  '''
  #calcolo l'accuracy
  accuracy = accuracy_score(y_true,y_pred)
  #precision, recall and f1 score con una media pesata (che permette di ottenere uno score più accurato per classi sbilanciate)
  precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  #creo il dizionario
  model_results = {"accuracy": accuracy,
                  "precision": precision,
                  "recall": recall,
                  "f1": f1_score}
  return model_results    