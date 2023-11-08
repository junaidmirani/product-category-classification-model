# product-category-classification-model



## Introduction

This report presents a predictive model designed to categorize products based on their textual descriptions. The model was developed to assist in automated product categorization, streamlining inventory management, and enhancing user experience in online retail and e-commerce platforms. The model leverages natural language processing techniques and machine learning algorithms to make accurate category predictions.

## Model Overview

### Data Preparation

- The training dataset consists of product descriptions and their corresponding categories, provided in the format: "MainCategory > SubCategory1 > SubCategory2 > ... > SubCategoryN."
- Text preprocessing was applied to the product descriptions, including text cleaning (removing punctuation), lowercasing, stop word removal, and word stemming using the Porter Stemmer.
- The categories were split into hierarchical levels, where the top-level category (MainCategory) represents the broad product category, and subsequent levels (SubCategories) provide more specific categorization.

### Feature Engineering

- The text data was transformed into numerical features using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This vectorization method captures the importance of words within each document while considering their frequency across the entire dataset.

### Model Training

- The training data was divided into a training set and a testing set using a 75-25 split ratio.
- A MiniBatchKMeans clustering algorithm was applied to the TF-IDF vectorized data to generate intermediate labels (auto-generated categories) based on the textual content.
- A LabelEncoder was used to encode the MainCategory labels.
- A logistic regression classifier was trained to predict the MainCategory based on the TF-IDF features.
- The model was saved for later use.

## Predictions

- The trained model can be used to predict the  subCategories of a product based on its description.

## Conclusion

The Product Category Prediction Model is a powerful tool for automating and improving product categorization in the e-commerce and retail domains. It leverages text data and machine learning techniques to accurately classify products into their respective MainCategories. Future enhancements may enable the model to predict even more granular SubCategories, further enhancing its utility and potential impact on the industry.

---
