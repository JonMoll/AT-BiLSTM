# AT-BiLSTM
### Requirements
1. Instalar requirements: `pip install -r requirements.txt`
2. Tambien instalar pytorch (>= 1.60) <br> 
Versión con CUDA 10.2: <br> 
`pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html` <br>
Versión con CPU: <br>
`pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

### Antes de entrenar el modelo
1. Colocar el archivo .csv de los datos en la carpeta `./data`
2. Generar el estado pre-entrenado del modelo FastText (skipgram) para los embeddings: <br>
`py ./utilities/train_fasttext/train.py skipgram` <br>
El estado se guardara en la carpeta `./states_fasttext`
3. Generar el dataset de entrenamiento y prueba: <br>
`py ./utilities/generate_datasets.py` <br>
Los datasets se guardarán en la carpeta `./states_fasttext`

### Entrenar el modelo
`py ./train.py`
* El archivo `./parameters.json` contiene los parametros de prueba y entrenamiento
* Los estados del modelo se guardarán en la carpeta `./states_models` a medida que es entrenado
* Al finalizar el entrenamiento se creara el archivo `./train_accuracy_f1.txt` que contiene la evolución del accuracy

### Probar el modelo
`py ./test.py estado_preentrenado`
* Reemplazar `estado_preentrenado` con el nombre de uno de los estados que contiene la carpeta `./states_models`, por ejemplo: <br>
`py ./test.py at-bilstm_epoch0_ac0.21973597359735975_f1score0.014086492469189213`
* Al finalizar el entrenamiento se creara el archivo `./test_accuracy_f1.txt` que contiene el accuracy del modelo con el dataset de prueba
