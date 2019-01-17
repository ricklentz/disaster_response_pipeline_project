# Disaster Response Pipeline Project

This project demonstrates several practices when increasing the value of unstructured text data.  The data is combined and cleaned during the first step of the pipeline.  Machine learning is applied to the processed data to build and test a model.  The model is used by the Flask web application to display simple distributions of the historical data and demonstrate the classification of new messages into each of the 36 categories.  


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
