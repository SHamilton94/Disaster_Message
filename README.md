# Disaster Response Pipeline Project

# Summary:
# The Disaster Response Pipeline Project helps to aid individuals who sent
# messages sent during various disasters. The ETL pipeline will clean the data, use 
# a predictive ML model to classify the type of disaster messages sent and received, 
# deploy to a web app, and provide data visualization 

#File Descriptions:
# process_data.py prepares and cleans the data to put it in a usable format for the ML model
# train_classifier.py trains the ML classification model
# run.py implements the classification model 
# master.html designs the web app
# go.html deploys the web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        Navigate to the data folder, then run the following
        `python process_data.py messages.csv categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
