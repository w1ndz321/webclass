from lib import *
from process import load_and_process_data
from url_data import process_urls
from tfidf import tfidf

train_data_dir = '../data/train/'
test_data_file = '../data/test/test.csv'

def label_encode(df):
    for col in tqdm(['method', 'refer', 'url_filetype', 'ua_short', 'ua_first']):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def run_lgb(df_train, df_test, use_features):
    target = TARGET
    oof_pred = np.zeros((len(df_train), NUM_CLASSES))
    y_pred = np.zeros((len(df_test), NUM_CLASSES))
    
    class_weights = compute_class_weight('balanced', classes=np.unique(df_train[target]), y=df_train[target])
    class_weight_dict = {i: class_weights[i] for i in range(NUM_CLASSES)}
    custom_weights={0: 0.204/0.422,1: 0.201/0.299,2: 0.199/0.195,3: 0.176/0.042,4: 0.116/0.020,5: 0.101/0.019}
    
    folds = StratifiedKFold(n_splits=10)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train[TARGET])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)
        params = {
            'learning_rate': 0.05,
            'metric': 'multiclass',
            'objective': 'multiclass',
            'num_classes': NUM_CLASSES,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.9,
            'min_data_in_leaf':100,
            'bagging_freq': 10,
            'n_jobs': -1,
            'nthread': -1,
            'max_depth': 8,
            'num_leaves': 31,
            'lambda_l1': 0.5,
            'lambda_l2': 0.8,
            'verbose': -1,
        }
        model = lgb.train(params, 
                          train_set, 
                          num_boost_round=500,
                          callbacks = [log_evaluation(100), early_stopping(stopping_rounds=60)],
                          valid_sets=[train_set, val_set],
                          )
        oof_pred[val_ind] = model.predict(x_val)
        y_pred += model.predict(df_test[use_features]) / folds.n_splits
        del x_train, x_val, y_train, y_val, train_set, val_set
        gc.collect()
    return y_pred, oof_pred

if __name__ == '__main__':
    NUM_CLASSES = 6
    TARGET = 'label'
    df = load_and_process_data(train_data_dir, test_data_file)
    df = process_urls(df)
    df = tfidf(df)
    df = label_encode(df)
    train = df[df['label'].notna()]
    test = df[df['label'].isna()]
    exclude_ = ['label', 'user_agent', 'body', 'url_unquote', 'id','url','url_query', 'url_path' ]
    features = [col for col in df.columns if col not in exclude_]
    y_pred, oof_pred = run_lgb(train, test, features)
    print(accuracy_score(np.argmax(oof_pred, axis=1), train['label']))
    sub = pd.read_csv('../data/submit_example.csv')
    sub['predict'] = np.argmax(y_pred, axis=1)
    sub.to_csv('results.csv', index=False)
    print('Generated submission file: results.csv!!!!!!!')


