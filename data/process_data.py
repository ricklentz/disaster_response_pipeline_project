import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
     """Combines the messages and categorical label csv files.

    arguments:
        messages_filepath -- the path to the messsages csv
        categories_filepath -- the path to the labels csv

    returns:
        combinded dataframe
    """
    
    cats = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)

     # add the categories
    for category in [x.split('-')[0] for x in cats.categories[0].split(';')]:
        # add the category column
        messages[category] = 0

    # populate the category values
    for index, row in messages.iterrows():
        message_id = row['id']
        category = cats.loc[cats['id'] == message_id].categories.iloc[0]
        rec_cats = [t.replace('-1','') for t in category.split(';') if '-1' in t]
        for category in [t.replace('-1','') for t in category.split(';') if '-1' in t]:
            messages.at[index,category] = 1
    return messages


def clean_data(df):
    """Removes duplicates of the message content, keeping the first instance of each unique message."""
    df.drop_duplicates(subset='message', keep='first', inplace=True)
    return df


def save_data(df, database_filename):
    """Writes the data frame to a sqlite database.

    Keyword arguments:
    df -- the pandas dataframe to be loaded into the sqlite table
    database_filename -- the path to the sqlite database file
    """
    conn = sqlite3.connect(database_filename)
    df.to_sql(name = 'messages',con = conn)


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
              ' to as the third argument. \n\n Example: \n'\
              ' python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()