import csv
import random
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
  def __init__(self):
    self.data = []
    self.labels = []
    self.max_num_words = 0

  # ======================================================================

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]

  # ======================================================================

  def ShuffleData(self):
    random.shuffle(self.data)

  # ======================================================================

  def ReadCSV(self, path):
    with open(path, 'r') as file_csv:
      data_csv = csv.reader(file_csv, delimiter=',')

      index_label = 0
      index_transcription = 1
      
      for row in data_csv:
        label = row[index_label]
        transcription = row[index_transcription]

        if not label in self.labels:
          self.labels.append(label)

        if len(transcription.split()) > self.max_num_words:
          self.max_num_words = len(transcription.split())

        self.data.append((transcription, label))

  # ======================================================================

  def BatchLabelsToBatchIndices(self, batch_labels):
    batch_indices = []

    for label in batch_labels:
      index = self.labels.index(label)
      batch_indices.append(index)
    
    batch_indices = np.array(batch_indices)

    return batch_indices
