import pandas as pd

def load_data(file_path):

    df = pd.read_csv(file_path)
    return df

file_path="data/credit_card_fraud.csv"

df=load_data(file_path)

print(df.head())