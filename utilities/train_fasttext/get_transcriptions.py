import csv

from stop_words import StopWords

class GetTranscriptions:
  def __init__(self):
    self.stop_words = StopWords()
    self.transcriptions = []
    self.transcriptions_NOstopwords = []

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
  Input:
    sentence = ' el perro  come  su comida'
  
  Output:
    return 'el perro come su comida'
  '''

  def DeleteExtraSeparations(self, sentence):
    words = sentence.split()
    separator = ' '

    return separator.join(words)

  # ======================================================================

  '''
    Input:
      path_data = './data.csv'
      
      data.csv:
        ,conid,sequence,etiqueta,transcription
        0,31321,0,label1,el perro come su comida
        1,15216,1,label2,el gato duerme
    
    Output:
      self.transcriptions = ['el perro come su comida', 'el gato duerme']
      self.transcriptions_NOstopwords = ['perro come comida', 'gato duerme']
  '''

  def ReadData(self, path_data):
    with open(path_data, 'r') as file_csv:
      data_csv = csv.reader(file_csv, delimiter=',')

      count_row = 0
      index_transcription = 4
      
      for row in data_csv:
        if count_row != 0:
          transcription = row[index_transcription]
          self.transcriptions.append(transcription)

          transcription_NOstopwords = self.stop_words.DeleteStopWords(row[index_transcription])
          self.transcriptions_NOstopwords.append(transcription_NOstopwords)

        count_row += 1

  # ======================================================================

  def WriteFile(self, path, transcriptions):
    with open(path, 'w') as file_txt:
      for transcription in transcriptions:
        transcription = self.DeleteExtraSeparations(transcription)

        if transcription != '':
          file_txt.write(transcription + '\n')

  # ======================================================================

  def ExportData(self):
    name_transcriptions = 'transcriptions.txt'
    path_transcriptions = './data/' + name_transcriptions
    self.WriteFile(path_transcriptions, self.transcriptions)

    name_transcriptions_NOstopwords = 'transcriptions_NOstopwords.txt'
    path_transcriptions_NOstopwords = './data/' + name_transcriptions_NOstopwords
    self.WriteFile(path_transcriptions_NOstopwords, self.transcriptions_NOstopwords)
