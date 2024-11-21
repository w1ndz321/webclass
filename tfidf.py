

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def add_tfidf_feats(df, col, n_components=16, analyzer='char', suffix=''):
    text = list(df[col].values)
    tf = TfidfVectorizer(min_df=5,
                         analyzer=analyzer,
                         ngram_range=(1, 5),
                         stop_words='english')
    tf.fit(text)
    X = tf.transform(text)

    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    X_svd = svd.transform(X)

    for i in range(n_components):
        df[f'{col}_tfidf{suffix}_{i}'] = X_svd[:, i]
    return df

def tfidf(df):
    df = add_tfidf_feats(df, 'url_unquote', n_components=16, analyzer='char', suffix='')
    df = add_tfidf_feats(df, 'user_agent', n_components=16, analyzer='char', suffix='')
    df = add_tfidf_feats(df, 'body', n_components=16, analyzer='char', suffix='')

    df = add_tfidf_feats(df, 'url_unquote', n_components=16, analyzer='word', suffix='_word')
    df = add_tfidf_feats(df, 'user_agent', n_components=16, analyzer='word', suffix='_word')
    df = add_tfidf_feats(df, 'body', n_components=256, analyzer='word', suffix='_word')

    df = add_tfidf_feats(df, 'url_unquote', n_components=16, analyzer='char', suffix='_char')
    df = add_tfidf_feats(df, 'user_agent', n_components=16, analyzer='char', suffix='_char')
    df = add_tfidf_feats(df, 'body', n_components=256, analyzer='char', suffix='_char')

    return df
