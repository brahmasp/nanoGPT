import numpy as np

# Filter out sentiment token from the input # 
def filter_sentiment_token(input_ids, sentiment_token):
    return np.array([x for x in input_ids if x != sentiment_token])