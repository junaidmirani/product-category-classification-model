

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import joblib
from joblib import load
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt')

stop = stopwords.words('english')
porter = PorterStemmer()

# Load the trained model, vectorizer, KMeans model, and label encoder
model = load('testing/category_classifier.joblib')
vectorizer = load('testing/tfidf_vectorizer.joblib')
kmeans_model = load('testing/kmeans_model.joblib')
label_encoder = load('testing/label_encoder.joblib')


def preprocess_data(text):
    ''' Applying stopwords and stemming on raw data'''
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


# Define a sample product description
description = input(" product description: ")

# Preprocess the description
description = preprocess_data(description)

# Vectorize the preprocessed description
description_features = vectorizer.transform([description])

# Predict the category
predicted_category = model.predict(description_features)

# Apply MiniBatchKMeans for auto-generated categories
auto_generated_labels = kmeans_model.predict(description_features)

# Now you can safely transform any new auto-generated labels
auto_generated_category_names = label_encoder.inverse_transform(
    auto_generated_labels)
#
#
#
# Load the classes
label_encoder_classes = joblib.load('testing/label_encoder_classes.joblib')

# Check if the new label is in the classes before transforming
if auto_generated_labels[0] in label_encoder_classes:
    auto_generated_category_names = label_encoder.inverse_transform(
        auto_generated_labels)
else:
    auto_generated_category_names = ['unknown']


print(f'Predicted Category: {predicted_category[0]}')
print(f'Auto Generated Category: {auto_generated_category_names[0]}')
