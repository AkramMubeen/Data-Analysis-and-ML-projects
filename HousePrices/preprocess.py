import pickle
import pandas as pd

def make_df(data):
    df = pd.DataFrame(data,index=[0])
    df['GaragePresence'] = df['GarageYrBlt'].notnull().astype(int)
    df = df.drop('GarageYrBlt', axis=1)
    Id = df.Id
    df = df.drop('Id',axis=1)
    df.replace("NA", "nan", inplace=True)
    return df

def load_and_apply_label_encoders(df, columns_to_encode, file_paths):
    # Load each label encoder using pickle
    label_encoders = {}
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            label_encoder = pickle.load(file)
        column_name = file_path.split("/")[-1].replace("label_", "").replace(".pkl", "")
        label_encoders[column_name] = label_encoder

    # Now you have a dictionary `label_encoders` where each key is the column name and the value is the corresponding label encoder object.
    for column in columns_to_encode:
        label_encoder = label_encoders[column]  # Get the corresponding label encoder for the column
        df[column] = label_encoder.transform(df[column].astype(str))

    return df

def load_and_apply_scaler(df,columns_to_scale,file_path):
    with open(file_path, 'rb') as file:
        scaler = pickle.load(file)
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    return df