import nltk
from nltk.corpus import stopwords

class StopWords:
  def __init__(self):
    nltk.download('stopwords') # Se usan los stop words del corpus de nltk
    self.stop_words = stopwords.words('spanish')
    self.ReplaceStopWordsTildes()

  # ======================================================================

  '''
    * Agrega stop words de una archivo

    Input:
      self.stop_words = [su, de]
      path = './stopwords.txt'
      
      stopwords.txt:
        el
        la
        ...

    Output:
      self.stop_words = [su, de, el, la, ...]
  '''

  def ReadFile(self, path):
    with open(path, 'r') as file_txt:
      for row in file_txt:
        word = row.replace('\n', '') # Eliminar saltos de linea
        self.stop_words.append(word)

  # ======================================================================

  '''
    * Quita las tildes de una palabra

    Input:
      word = 'él'
    
    Output:
      return 'el'
  '''

  def ReplaceTildes(self, word):
    replacements = ( ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u') )

    for a, b in replacements:
      word = word.replace(a, b)
      word = word.replace(a.upper(), b.upper())
    
    return word

  # ======================================================================

  '''
    * Quita las tildes de self.stop_word

    Input:
      self.stop_word = ['él', 'la', 'sú']
    
    Output:
      self.stop_word = ['el', 'la', 'su']
  '''

  def ReplaceStopWordsTildes(self):
    for i in range(len(self.stop_words)):
      self.stop_words[i] = self.ReplaceTildes(self.stop_words[i])

  # ======================================================================

  '''
    * Elimina los stop words de una oracion

    Input:
      sentence = 'el perro come su comida'
    
    Output:
      return 'perro come comida'
  '''

  def DeleteStopWords(self, sentence):
    words = sentence.split()

    for i in range(len(words)):
      if words[i] in self.stop_words:
        words[i] = ''
    
    words = [word for word in words if word != ''] # Eliminando elementos vacios ('')

    separator = ' '
    sentence = separator.join(words)

    return sentence
