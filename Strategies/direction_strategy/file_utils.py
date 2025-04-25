import pandas as pd




def load(file_path):


    data_type = file_path.split('/')[-1].split('.')[-1]

    if data_type =='csv':
        data = pd.read_csv(file_path)
    elif data_type == 'parquet':
        data = pd.read_parquet(file_path)
    elif data_type == 'feather':
        data = pd.read_feather(file_path)
    elif data_type == 'h5':
        data = pd.read_hdf(file_path)
    elif data_type == 'pickle':
        data = pd.read_pickle(file_path)
    else:
        raise ValueError(f'data_type {data_type} not supported')

    return data



def save(data,file_path):

    data_type = file_path.split('/')[-1].split('.')[-1]

    if data_type =='csv':
        data.to_csv(file_path,index=False)
    elif data_type == 'parquet':
        data.to_parquet(file_path,index=False)
    elif data_type == 'feather':
        data.to_feather(file_path)
    elif data_type == 'h5':
        data.to_hdf(file_path,key='data')
    elif data_type == 'pickle':
        data.to_pickle(file_path)
    else:
        raise ValueError(f'data_type {data_type} not supported')