from traceback import print_tb
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
import pandas as pd
import librosa

model = tf.keras.models.load_model(r'model_3classes.h5')
inv_map = {0:'_angry', 1:'_neutral', 2:'_sad'}

def prepare_audio(audio):
    with open('test.wav', mode='bx') as f:
        f.write(audio)

    data = pd.DataFrame(columns=['feature'])
    #sprcify actual path here
    X, sample_rate = librosa.load('test.wav', res_type='kaiser_fast',duration=3,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data.loc[0] = [feature]
    a=np.array(data.feature[0])
    b=np.zeros(259-len(data.feature[0]))
    final_array = np.concatenate((a, b), axis=0)
    a1 = np.expand_dims(final_array, axis=1)
    a2 = np.expand_dims(a1, axis=0)

    os.remove("test.wav") 
    return a2


def predict_result(a2):
    pred = inv_map[model.predict(a2).argmax()]
    print(pred)
    return pred


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_sound():
    print(request)
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    print(file)

    if not file:
        return

    audio_bytes = file.read()
    audio = prepare_audio(audio_bytes)
    
    return jsonify(prediction=predict_result(audio))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
