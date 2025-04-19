import os
import pandas as pd
import logging
import nltk
import string
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

#Ensure that "logs" directory exist
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)


#logging configuration
logger = logging.getLogger('pre_processing')
logger.setLevel('DEBUG')

#logs for console configuration
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#saving logs on .log file configuration
log_file_path = os.path.join(log_dir, 'pre_processing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    
    ps = PorterStemmer()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]

    return ' '.join(text)

def pre_process_df(df, text_column='text', target_column='target'):
    try:
        logger.debug('Start preprocessing for Dataframe')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        #Remove duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicate removed')


        #Apply text transformation to the specific text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s',e)
        raise

def main(text_column = 'text', target_column = 'target'):
    try:
        #Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        #Transformed data
        train_preprocessed_data = pre_process_df(train_data,text_column,target_column)
        test_preprocessed_data = pre_process_df(test_data,text_column,target_column)
        logger.debug('Transformation complete')

        #Store data inside the data/interim
        data_path = os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_preprocessed_data.to_csv(os.path.join(data_path, 'train_transform.csv'),index = False)
        test_preprocessed_data.to_csv(os.path.join(data_path, 'test_transform.csv'),index = False)

        logger.debug('Processed data saved to: %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s',e)
        raise
    except Exception as e:
        logger.error('Failed to complete the data transformation processes: %s',e)
        raise


if __name__ == '__main__':
    main()
    






