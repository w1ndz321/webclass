import warnings
warnings.simplefilter('ignore')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
import os
import re
import numpy as np
from urllib.parse import unquote, urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import gc
from lightgbm import log_evaluation, early_stopping
from sklearn.utils.class_weight import compute_class_weight