"""Reddit prediction model Flask App"""

from flask import Flask, render_template, request
import pickle
import pandas as pd

    
app = Flask(__name__)
bayes = pickle.load(open('naive_bayes.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route("/")
def hello():
    return render_template('home.html')

@app.route("/querytest", methods=['GET'])
def query_test():
    x = request.args.get("text")
    return render_template('predict.html', forum = x)


@app.route("/predict", methods=['POST'])
@app.route("/predict/<text>", methods=['GET'])
def predict(text=None):
    try:
        if request.method == 'POST':
            text = request.values['text']
    except KeyError:
        return ('''Bad request: value missing''')
    else:
        def model_prediction(text):
            
            new = tfidf.transform([text])
            
            predicter = bayes.predict(new)

            return predicter[0]
        prediction = model_prediction(text)
        return str(prediction)



@app.route("/about")
def preds():
    return render_template('about.html')
