import os
import pandas as pd


def load_and_process_data(train_data_dir, test_data_file):
    # 加载训练数据
    df_train = pd.DataFrame()
    train_files = []

    for filename in os.listdir(train_data_dir):
        if filename.endswith('.csv'):
            train_files.append(os.path.join(train_data_dir, filename))

    for filepath in train_files:
        df = pd.read_csv(filepath)
        df_train = pd.concat([df_train, df]).reset_index(drop=True)

    df_train.fillna('__NaN__', inplace=True)
    df_train = df_train.rename(columns={'lable': 'label'})


    df_test = pd.read_csv(test_data_file)
    df_test.fillna('__NaN__', inplace=True)


    df = pd.concat([df_train, df_test]).reset_index(drop=True)

    return df
