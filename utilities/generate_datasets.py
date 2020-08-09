import csv
import math
import random

from train_fasttext.stop_words import StopWords

class GenerateDatasets:
  def __init__(self):
    self.stop_words = StopWords()
    self.labels = []
    self.data = {}

  # ======================================================================

  '''
    * Agrega mas stop words desde un archivo

    Input:
      self.stop_words.stop_words = [de, su]
      path_stopwords = ./stopwords.txt
    
      stopwords.txt:
        el
        la
        ...

    Output:
      self.stop_words.stop_words = [de, su, el, la, ...]
  '''

  def AddStopWords(self, path_stopwords):
    self.stop_words.ReadFile(path_stopwords)

  # ======================================================================

  '''
    * Se eliminan los stop words de las oraciones

    Input:
      path_data = './file.csv'

      file.csv:
        ,conid,sequence,etiqueta,transcription
        0,1001001049O0190520,0,c_labelA, el perro come su comida
        1,1001001049O0190520,1,c_labelB,el gato juega
      
    Output:
      self.labels = ['c_labelA', 'c_labelB']
      self.data = {
        'c_labelA': ['perro come comida', ...],
        'c_labelB': ['gato juega', ...]
      }
  '''

  def ReadFile(self, path_data):
    with open(path_data, 'r') as file_csv:
      data_csv = csv.reader(file_csv, delimiter=',')

      count_row = 0
      index_label = 3
      index_transcription = 4
      
      for row in data_csv:
        if count_row != 0:
          label = row[index_label]
          transcription = row[index_transcription]
          transcription = self.stop_words.DeleteStopWords(transcription)

          if transcription != '':
            if not label in self.labels:
              self.labels.append(label)
              self.data[label] = []
            
            self.data[label].append(transcription)
          
        count_row += 1

  # ======================================================================

  '''
    * Elimina las oraciones repetidas por cada categoria

    Input:
      self.data = {
        'c_labelA': ['perro come comida', 'perro come comida', ...],
        'c_labelB': ['gato juega', 'gato juega', ...]
      }
    
    Output:
      self.data = {
        'c_labelA': ['perro come comida', ...],
        'c_labelB': ['gato juega', ...]
      }
  '''

  def DeleteDuplicates(self):
    for label in self.labels:
      self.data[label] = list(dict.fromkeys(self.data[label]))

  # ======================================================================

  '''
    * Escribe un archivo con los elementos de un conjunto de datos (array de pares etiqueta oracion)

    Input:
      path = './dataset.csv'
      dataset = [('label1', 'sentence1'), ('label2', 'sentence2'), ...]
    
    Output:
      dataset.csv:
        label1,sentence1
        label2,sentence2
        ...
  '''

  def WriteFile(self, path, dataset):
    with open(path, 'w') as file_dataset:
      for pair in dataset:
        label = pair[0]
        sentence = pair[1]

        file_dataset.write(label + ',' + sentence + '\n')

  # ======================================================================

  '''
    * De las oraciones por cada etiqueta se extrae el 25% (len_label / 4) como datos de entrenamiento

    Input:
      self.labels = ['c_labelA', 'c_labelB']
      self.data = {
        'c_labelA': ['perro come comida', ...],
        'c_labelB': ['gato juega', ...]
      }

    Output:
      dataset_train.csv:
        label1,sentence1
        label2,sentence2
        ...
      
      dataset_test.csv:
        label1,sentence1
        label2,sentence2
        ...
  '''

  def ExportTrainTestDatasets(self, path_dataset_train, path_dataset_test):
    dataset_train = [] # [('label1', 'sentence1'), ('label2', 'sentence2'), ...]
    dataset_test = [] # [('label1', 'sentence1'), ('label2', 'sentence2'), ...]

    for label in self.labels:
      random.shuffle(self.data[label]) # Se desordenan las oraciones

      len_label = len(self.data[label])
      len_test = math.ceil(len_label / 4)
      index_middle = len_label - len_test
      
      subset_train = self.data[label][:index_middle]
      subset_test = self.data[label][index_middle:]

      for sentence_train in subset_train:
        dataset_train.append((label, sentence_train))
      
      for sentence_test in subset_test:
        dataset_test.append((label, sentence_test))

    self.WriteFile(path_dataset_train + str(len(dataset_train)) + '.csv', dataset_train)
    self.WriteFile(path_dataset_test + str(len(dataset_test)) + '.csv', dataset_test)

# ======================================================================

generate_datasets = GenerateDatasets()

name_stopwords = 'stopwords.txt'
path_stopwords = './data/' + name_stopwords
generate_datasets.AddStopWords(path_stopwords)

name_data = 'TC_TLV.csv'
path_data = './data/' + name_data
generate_datasets.ReadFile(path_data)

generate_datasets.DeleteDuplicates()

name_dataset_train = 'dataset_train_len'
path_dataset_train = './data/' + name_dataset_train

name_dataset_test = 'dataset_test_len'
path_dataset_test = './data/' + name_dataset_test

generate_datasets.ExportTrainTestDatasets(path_dataset_train, path_dataset_test)
