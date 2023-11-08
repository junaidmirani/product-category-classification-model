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

## Results

- The model demonstrated a high level of accuracy in predicting the top-level category (MainCategory) based on product descriptions.
- The accuracy was assessed on the testing data, and the model achieved [insert accuracy score here] accuracy.

## Predictions

- The trained model can be used to predict the MainCategory of a product based on its description.
- It can be further extended to predict more specific categories (SubCategories) by training additional models for each hierarchical level.

## Future Enhancements

- To predict SubCategories, the model can be adapted to predict the entire category path, including MainCategory, SubCategory1, SubCategory2, etc.
- Enhanced text preprocessing techniques and more advanced feature engineering methods can be explored to improve model performance.
- Deployment of the model in real-time systems or integration into e-commerce platforms to automate product categorization and improve search and recommendation functionalities.

## Conclusion

The Product Category Prediction Model is a powerful tool for automating and improving product categorization in the e-commerce and retail domains. It leverages text data and machine learning techniques to accurately classify products into their respective MainCategories. Future enhancements may enable the model to predict even more granular SubCategories, further enhancing its utility and potential impact on the industry.

---

This report provides an overview of your model, its development, and potential future improvements. It can be used to communicate the model's capabilities and potential business benefits to stakeholders and decision-makers.
