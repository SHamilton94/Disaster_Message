import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import nltk
import sqlite3
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split



def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterMessage.db')
    df = pd.read_sql('SELECT * FROM DisasterMessage', engine)
    df = df.dropna()
    X = df['message'] 
    y = df.iloc[:, 4:]
    y = y.astype('float')


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return tokens


def build_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 42)
    pipeline2 = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        #('hashvec', HashingVectorizer(n_features=10)),
        #('feathash', FeatureHasher()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10, max_depth=45)))   
    ])
    pipeline2.fit(X_train, y_train)
    y_pred = pipeline2.predict(X_test)
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    parameters = {
        'clf__n_estimators': [300]
    }
    cv2 = GridSearchCV(pipeline2, param_grid=parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    for i in range(0, Y_test.shape[1]):
        print(classification_report(y_test[:i], y_pred[:i]))


def save_model(model, model_filepath):
    import pickle 
    filename = 'Disaster_Pipeline.sav'
    pickle.dump(pipeline2, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()