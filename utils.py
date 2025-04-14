import pickle

def yield_data(pickle_file_path="./data/monash/monash-df.pkl"):
    """
    Generator function to yield objects one at a time from a pickle file.
    
    Args:
        pickle_file_path (str): Path to the pickle file.
    
    Yields:
        dict: A dictionary with filename as key and (DataFrame, frequency, horizon, 
              contain_missing_values, contain_equal_length) as value.
    
    Raises:
        FileNotFoundError: If the pickle file doesn't exist.
        pickle.UnpicklingError: If the pickle file is corrupted.
    """
    try:
        with open(pickle_file_path, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                    yield obj
                except EOFError:
                    break
    except FileNotFoundError:
        raise FileNotFoundError(f"Pickle file not found: {pickle_file_path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError("Corrupted pickle file.")
