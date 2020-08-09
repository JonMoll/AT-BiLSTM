import sys
import fasttext

from get_transcriptions import GetTranscriptions

arguments = sys.argv
model_type = arguments[1]

get_transcriptions = GetTranscriptions()

path_stopwords = './data/stopwords.txt'
get_transcriptions.AddStopWords(path_stopwords)

path_data = './data/TC_TLV.csv'
get_transcriptions.ReadData(path_data)

get_transcriptions.ExportData()

# ======================================================================

path_corpus = './data/transcriptions.txt'
path_corpus_NOstopwords = './data/transcriptions_NOstopwords.txt'

learning_rate = 0.05
dim_vectors = 150

model = fasttext.train_unsupervised(path_corpus, model=model_type, lr=learning_rate, dim=dim_vectors)
model_NOstopwords = fasttext.train_unsupervised(path_corpus_NOstopwords, model=model_type, lr=learning_rate, dim=dim_vectors)

len_vocabulary = len(model.words)
len_vocabulary_NOstopwords = len(model_NOstopwords.words)

# ======================================================================

name_model_state = 'fasttext_' + model_type + '_voc' + str(len_vocabulary) + '_dim' + str(dim_vectors) + '.bin'
path_model_state = './states_fasttext/' + name_model_state

model.save_model(path_model_state)

name_model_state = 'fasttext_' + model_type + '_NOstopwords_voc' + str(len_vocabulary_NOstopwords) + '_dim' + str(dim_vectors) + '.bin'
path_model_state = './states_fasttext/' + name_model_state

model_NOstopwords.save_model(path_model_state)
