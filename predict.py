import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_tweet(tweet):
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    return "⚠️ Hate Speech Detected" if result == 1 else "✅ Not Hate Speech"

if __name__ == "__main__":
    sample = input("Enter a tweet to check for hate speech:\n")
    print(predict_tweet(sample))