from flask import Flask,render_template,request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
labels = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
    36: 'a_',
    37: 'b_',
    38: 'c_',
    39: 'd_',
    40: 'e_',
    41: 'f_',
    42: 'g_',
    43: 'h_',
    44: 'i_',
    45: 'j_',
    46: 'k_',
    47: 'l_',
    48: 'm_',
    49: 'n_',
    50: 'o_',
    51: 'p_',
    52: 'q_',
    53: 'r_',
    54: 's_',
    55: 't_',
    56: 'u_',
    57: 'v_',
    58: 'w_',
    59: 'x_',
    60: 'y_',
    61: 'z_'
}

image_size = (224, 224)
model = keras.models.load_model('VGG19_v1_09_0.809.h5')
app=Flask(__name__)
@app.route('/',methods=['GET'])
def hello():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    imageFile=request.files['imagefile']
    image_path="./images/"+imageFile.filename
    imageFile.save(image_path)
    img = load_img(image_path, target_size=(image_size))
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)
    pred = model.predict(X)
    predictionclass=labels[pred[0].argmax()]
    return render_template('index.html',prediction=predictionclass)
if __name__=='__main__':
    app.run(debug=True)