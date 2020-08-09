import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from utilities.dataset import Dataset
from utilities.fasttext import FastText
from utilities.accuracy import Accuracy
from models.ATBiLSTM import ATBiLSTM

# ======================================================================

file_parameters = open('./parameters.json')
parameters = json.load(file_parameters)

path_dataset_train = parameters['path_dataset_train']
path_dataset_test = parameters['path_dataset_test']
path_fasttext = parameters['path_fasttext_state']
batch_size = parameters['train_batch_size']
epochs = parameters['train_epochs']

# ======================================================================

dataset_train = Dataset()
dataset_test = Dataset()

dataset_train.ReadCSV(path_dataset_train)
dataset_test.ReadCSV(path_dataset_test)

dataset_train.ShuffleData()

dataloader = data.DataLoader(dataset_train, batch_size)

# ======================================================================

max_num_words = dataset_train.max_num_words

if dataset_test.max_num_words > max_num_words:
  max_num_words = dataset_test.max_num_words

print('max num words: ' + str(max_num_words))

# ======================================================================

labels_dim = len(dataset_train.labels)

if len(dataset_test.labels) > labels_dim:
  labels_dim = len(dataset_test.labels)

print('num labels: ' + str(labels_dim))

# ======================================================================

embeddings = FastText(path_fasttext)

# ======================================================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ' + str(device))

# ======================================================================

embeddings_dim = embeddings.vectors_dim
hidden_dim = 150

model = ATBiLSTM(embeddings_dim, hidden_dim, labels_dim, max_num_words)
model = model.to(device)

# ======================================================================

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

accuracy = Accuracy()

# ======================================================================

max_f1score = 0.0

all_accuracy = []
all_f1score = []

for epoch in range(epochs):
  print('epoch: ' + str(epoch))

  y_true = []
  y_pred = []

  for i, batch in enumerate(dataloader):
    sentences = batch[0]
    labels = batch[1]
    
    sentences = embeddings.BatchSentencesToBatchVectors(sentences, max_num_words)
    labels = dataset_train.BatchLabelsToBatchIndices(labels)

    sentences = torch.Tensor(sentences)
    sentences = sentences.to(device)
    labels = torch.LongTensor(labels)
    labels = labels.to(device)

    model.zero_grad()
    model_output = model(sentences) # model_output.shape = [batch_size, labels_dim]

    loss = loss_function(model_output, labels)
    loss.backward()
    optimizer.step()

    y_true.append(labels)
    y_pred.append(accuracy.BatchValuesToBatchIndices(model_output))

  # ======================================================================

  y_true = accuracy.JoinEpochY(y_true, tensor=True)
  y_pred = accuracy.JoinEpochY(y_pred, tensor=False)

  epoch_accuracy = accuracy.CalculateAccuracy(y_true, y_pred)
  epoch_f1score = accuracy.CalculateF1Score(y_true, y_pred)

  all_accuracy.append(epoch_accuracy)
  all_f1score.append(epoch_f1score)

  print('\taccuracy: ' + str(epoch_accuracy))
  print('\tf1score: ' + str(epoch_f1score))

  if epoch_f1score > max_f1score:
    max_f1score = epoch_f1score

    name_model_state = 'at-bilstm_epoch' + str(epoch) + '_ac' + str(epoch_accuracy) + '_f1score' + str(epoch_f1score)
    path_model_state = './states_models/' + name_model_state

    torch.save(model.state_dict(), path_model_state)

    print('\tmodel saved: ' + path_model_state)

# ======================================================================

name_all_accuracy_f1 = 'train_accuracy_f1.txt'
path_all_accuracy_f1 = './' + name_all_accuracy_f1

with open(path_all_accuracy_f1, 'w') as all_accuracy_f1:
  all_accuracy_f1.write('accuracy f1score' + '\n')

  for i in range(len(all_accuracy)):
    all_accuracy_f1.write(str(all_accuracy[i]) + ' ' + str(all_f1score[i]) + '\n')
