from utils import yield_data, prepare_time_series

i = 0
data_generator = yield_data()
for data in data_generator:
    j = 0
    series_df_generator = prepare_time_series(data['df'], data['freq'])
    for series_df in series_df_generator:
        print(f"Dataset {i}, Series {j}")
        j += 1
    print(f"Finished dataset {i}")
    i += 1