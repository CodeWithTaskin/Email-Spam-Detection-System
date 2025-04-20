import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


#Ensure that "logs" directory exist
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

#logs for console configuration
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#saving logs on .log file configuration
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to perser the CSV file from: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpated error occurred while loading the data: %s',e)
        raise


def apply_tfidf(train_data : pd.DataFrame, test_data : pd.DataFrame, max_featuer : int) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_featuer)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('TfIdf applied and data transformed')

        return train_df,test_df
    
    except Exception as e:
        logger.error('Error during the Tfidf transformation: %s',e)
        raise


def save_data(df : pd.DataFrame, file_path : str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('Save data to %s',file_path)

    except Exception as e:
        logger.error('Unexpated error occurred while saving the data: %s',e)
        raise

def main():
    try:
        max_feature = 100

        train_data = load_data('./data/interim/train_transform.csv')
        test_data = load_data('./data/interim/test_transform.csv')

        train_df, test_df = apply_tfidf(train_data,test_data,max_featuer=max_feature)

        save_data(train_df,os.path.join('./data','processed','train_tfidf.csv'))
        save_data(train_df,os.path.join('./data','processed','test_tfidf.csv'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering processes: %s',e)
        raise

if __name__ == '__main__':
    main()
