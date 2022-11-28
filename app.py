from flask import Flask, request, json, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

## Load the model
loaded_model = None
with open('basic_classifier.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
vextorizer = None
with open('count_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
## predict
prediction = loaded_model.predict(vectorizer.transform(["True news"]))[0]
## print the prediction
print(prediction)

## Create the app
app = Flask(__name__)
## Create the route
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    ## Get the data
    data=request.args['text']
    print(data)
    ## Predict
    prediction = loaded_model.predict(vectorizer.transform([data]))[0]
    ## Return the prediction
    return jsonify({'prediction': prediction})
