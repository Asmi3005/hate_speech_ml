import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler
import pickle
from preprocessing import clean_text

df=pd.read_csv("data/labeled_data.csv")

# labels 0=hate sppech  1=offensive  2=neither 

df['label_binary']=df['class'].apply(lambda x:1 if x==0 else 0)

df['clean_tweet']=df['tweet'].apply(clean_text)

vectorizer=TfidfVectorizer(ngram_range=(1,2),max_df=0.9,min_df=3,stop_words='english')
X=vectorizer.fit_transform(df['clean_tweet'])

ros=RandomOverSampler(sampling_strategy=0.5,random_state=42)
X_balanced,y_balanced=ros.fit_resample(X,df['label_binary'])

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

svm_model=LinearSVC(class_weight='balanced')
svm_model.fit(X_train,y_train)
svm_pred=svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nClassification Report:\n", classification_report(y_test, svm_pred))

with open('model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model & vectorizer saved successfully.")