
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from joblib import load
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('punkt')

MODELSPATH = 'testing/category_classifier.joblib'
TFIDF_VECTORIZER_PATH = 'testing/tfidf_vectorizer.joblib'
LABEL_ENCODER_PATH = 'testing/label_encoder.joblib'
DATAFILE = 'ifound_cat.csv'

stop = stopwords.words('english')
porter = PorterStemmer()

# Load the trained TF-IDF vectorizer
vectorizer = load(TFIDF_VECTORIZER_PATH)

# Load the trained label encoder
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# print(label_encoder.classes_)


def load_model():
    '''Loading pretrained model'''
    model = load(MODELSPATH)
    return model


def preprocess_data(text):
    ''' Applying stopwords and stemming on raw data'''
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop]
    words = [porter.stem(word) for word in words]
    return " ".join(words)


if __name__ == '__main__':
    data = pd.read_csv(DATAFILE)
    model = load_model()

    # Preprocess the description column
    data['Description'] = data['Description'].apply(preprocess_data)

    # Convert descriptions to TF-IDF features
    description_features = vectorizer.transform(data['Description'])

    # Make predictions using the input features
    predictions = model.predict(description_features).astype(int)

    # Transform numerical labels to category names
    predicted_category_names = label_encoder.inverse_transform(predictions)
    # print(predicted_category_names)
    # Add predicted category names to the DataFrame
    data['Predicted_Category'] = predicted_category_names

    # Save the data with predicted category names
    data.to_csv('result.csv', index=False)
