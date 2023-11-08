# code is created by  JUNAID AHMED MIRANI
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
nltk.download('stopwords')

stop = stopwords.words('english')
porter = PorterStemmer()


def preprocess_data(text):
    ''' The function to remove punctuation, stopwords, and apply stemming'''
    words = re.sub("[^a-zA-Z]", " ", text)
    words = [word.lower() for word in words.split()
             if word.lower() not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


# Read the data
df = pd.read_csv('ifound_cat.csv')
df['Description'] = df['Description'].apply(preprocess_data)

# Flattening and Encoding the Labels
df['MainCategory'] = df['Category'].apply(lambda x: x.split(' > ')[-1])
label_encoder = LabelEncoder()
df['EncodedCategory'] = label_encoder.fit_transform(df['MainCategory'])

# Split the data
X = df['Description']
y = df['EncodedCategory']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Vectorize the data
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Apply MiniBatchKMeans for faster clustering
kmeans = MiniBatchKMeans(n_clusters=18, random_state=42)
auto_generated_labels = kmeans.fit_predict(X_train)

# Create a new DataFrame for the auto-generated labels
auto_generated_df = pd.DataFrame(
    {'auto_generated_category': auto_generated_labels})
# code is created by *********JUNAID AHMED MIRANI******
# Concatenate it with the original DataFrame
df = pd.concat([df, auto_generated_df], axis=1)

# Fit the label encoder on all possible labels
kmeans_encoder = LabelEncoder()
all_possible_labels = np.arange(kmeans.n_clusters)
kmeans_encoder.fit(all_possible_labels)
# Train a classification model (e.g., Logistic Regression)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
#
#
# Maximum label that MiniBatchKMeans can generate
print(np.max(all_possible_labels))
# Maximum label that LabelEncoder was trained on
print(len(label_encoder.classes_))
# print(np.max(label_encoder.classes_))
#
# Save the model
joblib.dump(classifier, 'testing/category_classifier.joblib')
joblib.dump(vectorizer, 'testing/tfidf_vectorizer.joblib')
# Save the KMeans model and label encoder
joblib.dump(kmeans, 'testing/kmeans_model.joblib')
joblib.dump(kmeans_encoder.classes_, 'testing/kmeans_encoder_classes.joblib')
joblib.dump(label_encoder, 'testing/label_encoder.joblib')
# The line joblib.dump(label_encoder.classes_, 'testing/label_encoder_classes.joblib') should be placed
# in your train.py file, right after you fit the LabelEncoder.
#  This line is saving the classes of the LabelEncoder after it has been fit on the training data.
joblib.dump(label_encoder.classes_, 'testing/label_encoder_classes.joblib')

print(f'Model Accuracy: {score}')
