import os
import pandas as pd
import logging
import pickle
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

#Ensure that "logs" directory exist
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

#logs for console configuration
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#saving logs on .log file configuration
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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
        logger.debug('Data loaded from %s with shape %s',file_path, df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parser the csv file: %s',e)
        raise
    
    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s',e)
        raise
    
def load_model(file_path : str) -> None:
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s',file_path)
        return model
    
    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s',e)
        raise

def evaluate_model(clf, X_test : np.ndarray, y_test : np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,-1]
        
        accuracy = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)
        
        matrix = {
            'accuracy' : accuracy,
            'recall': recall,
            'precision': precision,
            'auc': auc
        }
        
        logger.debug('Model evaluation matrix calculated')
        return matrix
    
    except Exception as e:
        logger.error('Unexpected error occurred while model evaluation: %s',e)
        raise
    
def save_data(file_path : str, matrix : dict) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path,'w') as file:
            json.dump(matrix, file, indent=4) 
        logger.debug('Matrix saved to %s',file_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving metrix: %s',e)
        raise
    

def main():
    try:
        clf = load_model('./model/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        matrix = evaluate_model(clf=clf, X_test=X_test, y_test=y_test)
        
        save_data(matrix=matrix,file_path='reports/matrix.json')
        
    except Exception as e:
        logger.error('Unexpected error occurred while saving matrix: %s',e)
        raise
    
if __name__ == '__main__':
    main()