import numpy as np
import pandas as pd
import yaml
import os 
from sklearn.model_selection import train_test_split 
import logging 

import logging
from logging import StreamHandler

# logging configure 
logger = logging.getLogger('data_ingestion')
console_handler = StreamHandler()
logger.setLevel(logging.DEBUG)

#file handeller 
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
#file formater
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            test_size = yaml.safe_load(file)['data_ingestion']['test_size']
        return test_size
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f"Error reading data from URL: {str(e)}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df.replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def save_data(data_path: str, training_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        training_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.info(f"Data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

def main() -> None:
    try:
        test_size = load_params('params.yaml')
        logger.info(f"Loaded test size: {test_size}")

        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        final_df = process_data(df)
        logger.info(f"Data processed. Final shape: {final_df.shape}")

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logger.info(f"Data split. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        data_path = os.path.join('data', 'raw')
        save_data(data_path, train_data, test_data)
        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.critical(f"An error occurred during data ingestion: {str(e)}")

if __name__ == '__main__':
    main()