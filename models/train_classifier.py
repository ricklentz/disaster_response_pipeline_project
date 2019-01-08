import sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlite3
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib

def load_data(database_filepath):
    print(database_filepath)
    conn = sqlite3.connect(database_filepath)
    
    results = pd.read_sql_query("SELECT * FROM 'messages';", conn)
    return results['message'], results[['related','request','offer','aid_related','medical_help','medical_products','search_and_rescue','security','military','child_alone','water','food','shelter','clothing','money','missing_people','refugees','death','other_aid','infrastructure_related','transport','buildings','electricity','tools','hospitals','shops','aid_centers','other_infrastructure','weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report']],['related','request','offer','aid_related','medical_help','medical_products','search_and_rescue','security','military','child_alone','water','food','shelter','clothing','money','missing_people','refugees','death','other_aid','infrastructure_related','transport','buildings','electricity','tools','hospitals','shops','aid_centers','other_infrastructure','weather_related','floods','storm','fire','earthquake','cold','other_weather','direct_report']

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    return RandomForestClassifier()



def evaluate_model(model, X_test, Y_test, category_names,vect, tfidf):
    y_pred = model.predict(tfidf.transform(vect.transform(X_test)))
    # display results
    #display_results(y_test, y_pred)


def save_model(model, model_filepath):
    _ = joblib.dump(model, model_filepath, compress=9)


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
        vect = CountVectorizer(tokenizer=tokenize)
        tfidf = TfidfTransformer()
        X_train_counts = vect.fit_transform(X_train)
        X_train_tfidf = tfidf.fit_transform(X_train_counts)
        model.fit(X_train_tfidf, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, vect, tfidf)

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