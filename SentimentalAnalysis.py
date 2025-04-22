import pandas as pd
import nltk
import tkinter as tk
from tkinter import messagebox, scrolledtext
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('RawData.csv')
df.columns = df.columns.str.strip()

# Preprocess text data
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['processed_review'] = df['Review'].apply(preprocess)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['Sentiment'], test_size=0.2, random_state=42)

# Naive Bayes classifier setup
def get_features(tokens):
    return {word: True for word in tokens}
train_set_nb = [(get_features(word_tokenize(review)), sentiment) for review, sentiment in zip(X_train, y_train)]
classifier_nb = NaiveBayesClassifier.train(train_set_nb)
test_set_nb = [(get_features(word_tokenize(review)), sentiment) for review, sentiment in zip(X_test, y_test)]

# SVM classifier setup
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
classifier_svm = SVC(kernel='linear', probability=True, random_state=42)
classifier_svm.fit(X_train_count, y_train)
y_pred_svm = classifier_svm.predict(X_test_count)

# Prediction function
def predict_sentiment(review):
    review_processed = preprocess(review)
    review_count = vectorizer.transform([review_processed])

    nb_sentiment = classifier_nb.classify(get_features(word_tokenize(review_processed)))
    nb_confidence = classifier_nb.prob_classify(get_features(word_tokenize(review_processed))).prob(nb_sentiment)

    svm_sentiment = classifier_svm.predict(review_count)[0]
    svm_confidence = classifier_svm.predict_proba(review_count).max()

    return {
        "Naive Bayes": (nb_sentiment, nb_confidence),
        "SVM": (svm_sentiment, svm_confidence)
    }

# Tkinter GUI
app = tk.Tk()
app.title("Sentiment Analyzer")
app.geometry("700x450")
app.configure(bg="#2c3e50")

# Title
title_label = tk.Label(app, text="Sentiment Analyzer", font=("Arial", 24, "bold"), fg="#ecf0f1", bg="#2c3e50")
title_label.pack(pady=15)

# Input frame
input_frame = tk.Frame(app, bg="#34495e", padx=10, pady=10)
input_frame.pack(padx=20, pady=10, fill="x")
# Neural Network setup for lemmatization

input_label = tk.Label(input_frame, text="Enter your review:", font=("Arial", 14), fg="#ecf0f1", bg="#34495e")
input_label.pack(anchor="w")

entry = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=80, height=6, font=("Arial", 12))
entry.pack(pady=5)

# Result frame
result_frame = tk.Frame(app, bg="#34495e", padx=10, pady=10)
result_frame.pack(padx=20, pady=10, fill="both", expand=True)

result_label = tk.Label(result_frame, text="Prediction Results:", font=("Arial", 16, "bold"), fg="#ecf0f1", bg="#34495e")
result_label.pack(anchor="w")

result_text = tk.Text(result_frame, height=8, font=("Arial", 12), bg="#ecf0f1", fg="#2c3e50", state="disabled", bd=2, relief="sunken")
result_text.pack(fill="both", expand=True, pady=5)

# Predict button
def on_predict():
    review = entry.get("1.0", tk.END).strip()
    if not review:
        messagebox.showwarning("Input Error", "Please enter a review.")
        return

    predictions = predict_sentiment(review)
    result_str = (
        f"Naive Bayes:\n  Predicted Sentiment: {predictions['Naive Bayes'][0]}\n  Confidence: {predictions['Naive Bayes'][1]:.2f}\n\n"
        f"SVM:\n  Predicted Sentiment: {predictions['SVM'][0]}\n  Confidence: {predictions['SVM'][1]:.2f}"
    )
    result_text.config(state="normal")
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, result_str)
    result_text.config(state="disabled")

button = tk.Button(app, text="Predict Sentiment", command=on_predict, font=("Arial", 14), bg="#27ae60", fg="white", padx=15, pady=8)
button.pack(pady=15)

app.mainloop()
