
import re
import numpy as np
from urllib.parse import urlparse, unquote

def get_url_query(s):
    li = re.split('[=&]', urlparse(s)[4])
    return [li[i] for i in range(len(li)) if i % 2 == 1]

def find_max_str_length(x):
    li = [len(i) for i in x]
    return max(li) if len(li) > 0 else 0

def find_str_length_std(x):
    li = [len(i) for i in x]
    return np.std(li) if len(li) > 0 else -1

def find_url_filetype(x):
    try:
        return re.search(r'\.[a-z]+', x).group()
    except:
        return '__NaN__'

def process_urls(df):
    df['url_unquote'] = df['url'].apply(unquote)
    df['url_query'] = df['url_unquote'].apply(lambda x: get_url_query(x))
    df['url_query_num'] = df['url_query'].apply(len)
    df['url_query_max_len'] = df['url_query'].apply(find_max_str_length)
    df['url_query_len_std'] = df['url_query'].apply(find_str_length_std)

    df['url_path'] = df['url_unquote'].apply(lambda x: urlparse(x)[2])
    df['url_filetype'] = df['url_path'].apply(find_url_filetype)
    df['url_path_len'] = df['url_path'].apply(len)
    df['url_path_num'] = df['url_path'].apply(lambda x: len(re.findall('/', x)))

    df['ua_short'] = df['user_agent'].apply(lambda x: x.split('/')[0])
    df['ua_first'] = df['user_agent'].apply(lambda x: x.split(' ')[0])

    # 检测危险关键词
    dangerous_keywords = ['select', 'union', 'delete', 'drop', 'update', 'exec', 'script']
    df['url_dangerous_keywords'] = df['url_unquote'].apply(lambda x: sum(1 for keyword in dangerous_keywords if keyword.lower() in x.lower()))

    df['url_params_count'] = df['url'].apply(lambda x: len(urlparse(x).query.split('&')))
    df['body_length'] = df['body'].str.len()
    df['is_json'] = df['body'].apply(lambda x: 1 if x.strip().startswith('{') else 0)
    df['body_special_char_ratio'] = df['body'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)) / (len(x) + 1))

    return df
