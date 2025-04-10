from flask import Flask, render_template,request
import pickle
import os

model=pickle.load(open("model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    tweet_vector = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vector)[0]
    
    if prediction == 1:
        result = "Yes, it's hate speech ❌"
    else:
        result = "No, it's not hate speech ✅"

    return render_template('index.html', tweet=tweet, result=result)

if __name__ == '__main__':
    app.run(debug=True)
