import sys
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlite3
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
 

def load_data(database_filepath):
    """Loads data from a sqlite database file.

    params:
        database_filepath -- relative file path

    returns:
        tuple of results and labels
    """
    conn = sqlite3.connect(database_filepath)
    
    results = pd.read_sql_query("SELECT * FROM 'messages';", conn)
    return results['message'], results[['related','request','offer','aid_related','medical_help','medical_products','search_and_rescue','security','military','child_alone','water','food','shelter','clothing','money','missing_people','refugees','death','other_aid','infrastructure_related','transport','buildings','electricity','tools','hospitals','shops','aid_centers','other_infrastructure','weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report']],['related','request','offer','aid_related','medical_help','medical_products','search_and_rescue','security','military','child_alone','water','food','shelter','clothing','money','missing_people','refugees','death','other_aid','infrastructure_related','transport','buildings','electricity','tools','hospitals','shops','aid_centers','other_infrastructure','weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report']

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', RandomForestClassifier())
    ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 1.0),#.75
        #'features__text_pipeline__vect__max_features': (None, 10000),#5000
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__n_estimators': [100, 200],#50
        'clf__min_samples_split': [2,4]#3
        #'features__transformer_weights': (
        #    {'text_pipeline': 1},
        #    {'text_pipeline': 0.5},
        #    {'text_pipeline': 0.8},
        #)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:,i],y_preds[:,i]))


def save_model(model, model_filepath):
    _ = joblib.dump(model, model_filepath, compress=9)

def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        # train classifier
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: '\
              'python train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()