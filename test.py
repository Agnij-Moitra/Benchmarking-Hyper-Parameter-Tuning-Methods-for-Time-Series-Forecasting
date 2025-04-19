from joblib import Parallel, delayed
from utils import *


def process_time_series(pickle_file_path="./data/monash/monash-df.pkl"):
    for data in yield_data(pickle_file_path):
        name, df, freq = data['name'], data['df'], data['freq']
        print(f"Processing {name} with frequency {freq}")
        for ts_df in prepare_time_series(df, freq):
            featurized_df = auto_featurize(ts_df, freq)
            print(
                f"Featurized DataFrame for {ts_df.columns[0]}:\n{featurized_df.head()}")
