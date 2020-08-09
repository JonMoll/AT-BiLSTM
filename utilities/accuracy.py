from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class Accuracy:
  '''
    Input:
      y_true = [1, 2, 0, 1]
      y_true = [0, 1, 2, 1]
    
    Output:
      return 0.1
  '''

  def CalculateAccuracy(self, y_true, y_pred):
    return accuracy_score(y_true, y_pred)

  # ======================================================================

  '''
    Input:
      y_true = [1, 2, 0, 1]
      y_true = [0, 1, 2, 1]
    
    Output:
      return 0.1
  '''

  def CalculateF1Score(self, y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

  # ======================================================================

  '''
    Input:
      epoch_y = [[1, 2], [0, 1]]
    
    Output:
      return [1, 2, 0, 1]
  '''

  def JoinEpochY(self, epoch_y, tensor):
    new_y = []

    if tensor:
      for batch_y in epoch_y:
        for y in batch_y:
          new_y.append(y.item())
    
    else:
      for batch_y in epoch_y:
        for y in batch_y:
          new_y.append(y)

    return new_y
  
  # ======================================================================

  '''
    Input:
      values = [0.1, 0.5, 0.2]
    
    Output:
      return 1
  '''

  def MaxIndex(self, values):
    max_index = 0
    max_value = values[max_index]

    for i in range(len(values)):
      if values[i] > max_value:
        max_index = i
        max_value = values[i]
    
    return max_index

  # ======================================================================

  '''
    * Extra los indices de los elementos maximos por cada vector del batch

    Input:
      batch_values = [[0.1, 0.5, 0.2], [0.7, 0.1, 0.5]]
    
    Output:
      return [1, 0]
  '''

  def BatchValuesToBatchIndices(self, batch_values):
    max_indices = []
    
    for values in batch_values:
      max_index = self.MaxIndex(values)
      max_indices.append(max_index)
    
    return max_indices
