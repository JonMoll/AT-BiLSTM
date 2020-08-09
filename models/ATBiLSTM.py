import torch
import torch.nn as nn
import torch.nn.functional as F

class ATBiLSTM(nn.Module):
  def __init__(self, embeddings_dim, hidden_dim, labels_dim, max_num_words):
    super(ATBiLSTM, self).__init__()

    self.embeddings_dim = embeddings_dim
    self.hidden_dim = hidden_dim
    self.labels_dim = labels_dim
    self.max_num_words = max_num_words
    
    self.bilstm = nn.LSTM(input_size=self.embeddings_dim,
                          hidden_size=self.hidden_dim,
                          batch_first=True,
                          bidirectional=True)

    self.classification = nn.Linear(in_features=max_num_words, 
                                    out_features=self.labels_dim)

  # ======================================================================

  '''
    labels_dim = 2
    max_num_words = 12
    embeddings_dim = 150
    batch_size = 4
    hidden_dim = 4
    
    sentences_embeddings.shape = [4, 12, 150]
    H.shape = [4, 12, 8]
    beta.shape = [4, 12, 8]
    beta_mean.shape = [4, 12, 1]
    beta_mean.shape (squeeze) = [4, 12]
    y.shape = [4, 2]
  '''

  def forward(self, sentences_embeddings):
    H, _ = self.bilstm(sentences_embeddings)
    beta = F.softmax(H, dim=1) # Sotmax sobre las columnas
    beta_mean = beta.mean(dim=2, keepdim=True) # Media sobre las filas
    beta_mean = beta_mean.squeeze(-1) # Eliminando dimension extra: [[0.2],[0.1],[0.3]] -> squeeze -> [0.2,0.1,0.3]
    y = self.classification(beta_mean)

    return y
