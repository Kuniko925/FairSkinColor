import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_four_dataset(df):
    column = "labels"
    df_train, dummy_df = train_test_split(df, train_size=0.6, shuffle=True, random_state=42, stratify=df[column])
    df_valid, df_test = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=42, stratify=dummy_df[column])
    df_train, df_sample = train_test_split(df_train, train_size=0.6, shuffle=True, random_state=42, stratify=df_train[column])
    
    print('Final sizes - train:', len(df_train), 'validation:', len(df_valid), 'test:', len(df_test), 'sample:', len(df_sample))

    print("---sample-------------")
    print(df_sample.groupby("labels").size())
    print("---train-------------")
    print(df_train.groupby("labels").size())
    print("---valid-------------")
    print(df_valid.groupby("labels").size())
    print("---test-------------")
    print(df_test.groupby("labels").size())
    
    return df_train, df_valid, df_test, df_sample

def split_three_dataset(df):
    column = "labels"
    df_train, df_dummy = train_test_split(df, train_size=0.6, shuffle=True, random_state=42, stratify=df[column])
    df_valid, df_test = train_test_split(df_dummy, train_size=0.5, shuffle=True, random_state=42, stratify=df_dummy[column])
    
    print('Final sizes - train:', len(df_train), 'validation:', len(df_valid), 'test:', len(df_test))
    print("---train-------------")
    print(df_train.groupby("labels").size())
    print("---valid-------------")
    print(df_valid.groupby("labels").size())
    print("---test-------------")
    print(df_test.groupby("labels").size())
    
    return df_train, df_valid, df_test


def load_and_process_csv(file_path, prediction=False):
    df = pd.read_csv(file_path)
    df["labels"] = df["labels"].astype(str)
    df["skin tone"] = df["skin tone"].astype(str)
    if prediction==True:
        df["predictions"] = df["predictions"].astype(str)
    return df