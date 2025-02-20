import os
import sys
import pickle
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

from src.exception import CustomException

# Preprocess text data for machine learning models
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Remove non-alphanumeric characters
    return text

# Save object (model, vectorizer, etc.) to a file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Load object (model, vectorizer, etc.) from a file
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Evaluate multiple models and return the performance metrics
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

# Train a Naive Bayes model for text classification
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Train a Support Vector Machine (SVM) for text classification
def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

# Train a BERT model for text classification using HuggingFace
def train_bert(data, label_column, text_column):
    try:
        # Tokenize the text data for BERT
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_function(examples):
            return tokenizer(examples[text_column], padding="max_length", truncation=True)

        # Prepare the dataset for Hugging Face's Trainer API
        dataset = Dataset.from_pandas(data[[text_column, label_column]])
        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.rename_column(label_column, "label")

        # Load pre-trained BERT model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(data[label_column].unique()))

        # Split data into train and test
        train_dataset = dataset.shuffle().select([i for i in range(80)])  # First 80% for training
        test_dataset = dataset.shuffle().select([i for i in range(80, 100)])  # Remaining 20% for testing

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',          
            num_train_epochs=3,              
            per_device_train_batch_size=8,  
            per_device_eval_batch_size=16,  
            warmup_steps=500,               
            weight_decay=0.01,              
            logging_dir='./logs',           
        )

        # Set up the Trainer API
        trainer = Trainer(
            model=model,                         
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=test_dataset             
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        results = trainer.evaluate()
        return results

    except Exception as e:
        raise CustomException(e, sys)

# Vectorize the text data for training (using TF-IDF)
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer
