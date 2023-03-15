import flask
from flask import Flask,render_template,url_for,request
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf



#Initialize the useless part of the base64 encoded image.
init_Base64 = 21;

#Our dictionary
#label_dict = {0:'Apple', 1:'Banana', 2:'Pear', 3:'Pineapple', 4:'Strawberry'}
label_dict = {0:'bird', 1:'butterfly', 2:'cat', 3:'Pineapple', 4:'Strawberry'}


#Initializing the Default Graph (prevent errors)
##graph = tf.compat.v1.get_default_graph()

# Use pickle to load in the pre-trained model.
#with open(f'5_Fruit_Model/5_Fruit_Classification_model.pkl', 'rb') as f:
#        model = pickle.load(f)

model = pickle.load(open('5_Fruit_Model/bird_butterfly_cat_model.pkl', 'rb'))

@tf.function
def predict_with_eager_mode(inputs):
    return model.predict(inputs)

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')



#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        #global graph
        #with graph.as_default():
    if request.method == 'POST':
            
        final_pred = None
        #Preprocess the image : set the image to 28x28 shape
        #Access the image
        draw = request.form['url']
        #Removing the useless part of the url.
        draw = draw[init_Base64:]
        #Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #Resizing and reshaping to keep the ratio.
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        
        vect = vect.reshape(1,28, 28, 1).astype('float32')
        print('herehar')       
                #Launch prediction
        # tf.config.run_functions_eagerly(True)
        #model.compile(run_eagerly = True)
        my_prediction = model.predict(vect)
        #tf.config.run_functions_eagerly(False)
        print('herehar8')    
        print(my_prediction)
        print('herehar9')  
                    #Getting the index of the maximum prediction
        index = np.argmax(my_prediction[0])
                    #Associating the index and its value within the dictionnary
        final_pred = label_dict[index]

    return render_template('results.html', prediction =final_pred)
    #return render_template('results.html', prediction ='ban')


if __name__ == '__main__':
	app.run(debug=True)