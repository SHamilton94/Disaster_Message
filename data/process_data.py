import sys
import pandas as pd
import re
import numpy as np


def load_data(messages_filepath, categories_filepath):
	#load messages dataset
    messages = pd.read_csv('messages.csv')
	messages.head()
	# load categories dataset
	categories = pd.read_csv('categories.csv')
	categories.head()


def clean_data(df):
    # merge datasets
	df = messages.merge(categories, how = 'inner', on = 'id')
	df.head()
	# create a dataframe of the 36 individual category columns
	categories = df['categories'].str.split(";", expand = True)
	categories.head()
	# select the first row of the categories dataframe
	row = categories.iloc[0]
	row = row.str.split('-', expand = True)


	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything 
	# up to the second to last character of each string with slicing
	category_colnames = list(row[0])
	print(category_colnames)
	
	#print(row)
	# rename the columns of `categories`
	categories.columns = category_colnames
	categories.head()
	
	for column in categories:
    	# set each value to be the last character of the string
    	categories[column] = categories[column].apply(lambda x: x.split('-')[1] )
    	# convert column from string to numeric
    	categories[column] = categories[column].astype('int64')
	categories.head()
	
	categories.drop(categories.index[categories['related'] == 2], inplace = True)
	categories.related.value_counts()
	
	# drop the original categories column from `df`
	df = df.drop(columns = ['categories'])

	df.head()
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories], axis = 1)
	df.head()
	# check number of duplicates
	dups = len(df[df.duplicated()])
	print(dups)
	# drop duplicates
	df.drop_duplicates(inplace = True)
	# check number of duplicates
	print(len(df[df.duplicated()]))


def save_data(df, database_filename):
	#saves database
    from sqlalchemy import create_engine
	engine = create_engine('sqlite:///DisasterMessage.db')
	df.to_sql('DisasterMessage', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()