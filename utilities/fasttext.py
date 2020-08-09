import fasttext
import numpy as np

class FastText:
  def __init__(self, path_state):
    self.model = fasttext.load_model(path_state) # Se carga un modelo pre-entrenado
    self.vectors_dim = self.model.get_dimension()
  
  # ======================================================================

  '''
    Input:
      max_num_words = 3
      vectors = [[0.1, 0.0, 0.2], 
                 [0.3, 0.1, 0.2]]
    
    Output:
      return [[0.1, 0.0, 0.2], 
              [0.3, 0.1, 0.2],
              [0.0, 0.0, 0.0]]
  '''

  def PushVectorZeros(self, vectors, max_num_words):
    vector_zeros = np.zeros(self.vectors_dim)
    num_vectors_zeros = max_num_words - len(vectors)

    for _ in range(num_vectors_zeros):
      vectors.append(vector_zeros)

    return vectors

  # ======================================================================

  '''
    * Retorna los vectores que representan las palabras de la oracion
    * Agrega vectores de ceros para igualar el tamaño de todas las oraciones

    Input:
      max_num_words = 3
      sentence = 'la manzana'
    
    Output:
      return [[0.1, 0.0, 0.2], 
              [0.3, 0.1, 0.2],
              [0.0, 0.0, 0.0]]
  '''

  def SentenceToVectors(self, sentence, max_num_words):
    vectors = []

    for word in sentence.split():
      vector = self.model.get_word_vector(word)
      vectors.append(vector)

    vectors = self.PushVectorZeros(vectors, max_num_words)
    vectors = np.array(vectors)

    return vectors

  # ======================================================================

  '''
    * Retorna los vectores que representan las palabras que conformas las oraciones de batch

    Input:
      batch_sentences = ['la manzana', 'el perro']
      max_num_words = 3

    Output:
      return [[[0.1, 0.0, 0.2], 
               [0.3, 0.1, 0.2],
               [0.0, 0.0, 0.0]],
              [[0.1, 0.0, 0.2], 
               [0.3, 0.1, 0.2],
               [0.0, 0.0, 0.0]]]
  '''

  def BatchSentencesToBatchVectors(self, batch_sentences, max_num_words):
    batch_vectors = []

    for sentence in batch_sentences:
      vectors = self.SentenceToVectors(sentence, max_num_words)
      batch_vectors.append(vectors)
    
    batch_vectors = np.array(batch_vectors)

    return batch_vectors
