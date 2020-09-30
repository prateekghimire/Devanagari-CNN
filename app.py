import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
mappingdict={
    0:'क',
    1:'ख',
    2:'ग',
    3:'घ',
    4:'ङ',
    5:'च',
    6:'छ',
    7:'ज',
    8:'झ',
    9:'ञ',
    10:'ट',
    11:'ठ',
    12:'ड',
    13:'ढ',
    14:'ण',
    15:'त',
    16:'थ',
    17:'द',
    18:'ध',
    19:'न',
    20:'प',
    21:'फ',
    22:'ब',
    23:'भ',
    24:'म',
    25:'य',
    26:'र',
    27:'ल',
    28:'व',
    29:'श',
    30:'ष',
    31:'स',
    32:'ह',
    33:'क्ष',
    34:'त्र',
    35:'ज्ञ',
    36:'०',
    37:'१',
    38:'२',
    39:'३',
    40:'४',
    41:'५',
    42:'६',
    43:'७',
    44:'८',
    45:'९'    
}

st.set_option('deprecation.showfileUploaderEncoding', False)

model=tf.keras.models.load_model('model.h5')

st.write("""
        # DEVANAGARI CHARACTER PREDICTION
""")

st.write("""
            ## Developed by Pratik Ghimire
""")

file=st.file_uploader("Please upload image of a single character ", type=['jpg','jpeg','png'])


def import_and_predict(image_data,model):
    image=ImageOps.fit(image_data,(32,32),Image.ANTIALIAS)
    test_img=np.asarray(image)
    test_img=np.invert(test_img)
    test_img=cv2.cvtColor(test_img,cv2.COLOR_BGRA2GRAY)
    test_img=np.array(test_img)/255.0
    test_img=cv2.resize(test_img,(32,32))
    image=np.reshape(test_img,(1,32,32,1))
    prediction=model.predict(image)
    result=prediction.argmax()
    confidence=round(prediction[0][prediction.argmax()]*100,2)
    return result,confidence 

if file is None:
    #st.text("Please upload an image ")
    pass
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction,confidence=import_and_predict(image,model)

    st.write('The machine predicted the character as '+str(mappingdict[prediction])+' with confidence of '+str(confidence)+'%')


