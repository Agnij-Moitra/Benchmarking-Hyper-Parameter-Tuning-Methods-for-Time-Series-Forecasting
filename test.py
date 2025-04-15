from utils import yield_data, prepare_time_series

i = 0
data_generator = yield_data()
for data in data_generator:
    prepare_time_series(data['df'], data['freq'])
    print(i)
    i += 1