import logging
import os
import re

import numpy as np

from config.config import CONTRACT_DICT

def clean_words(x_series_):
    
    c_re_ = re.compile('(%s)' % '|'.join(CONTRACT_DICT.keys()))

    def expand_contractions(text, c_re=c_re_):
        def replace(match):
            return contractions_list[match.group(0)]

        return c_re.sub(replace, text)

    x_ = [re.sub('[^0-9a-z\' ]+', ' ', item.lower()).split() for item in x_series_]

    processed_x = []
    for eachQuery in x_:
        query = []
        for eachToken in eachQuery:
            expanded_token = expand_contractions(eachToken)
            # Applied to normalize words like word's, mother's, 1980's etc.
            if '\'' in expanded_token:
                expanded_token = expanded_token.strip('\'s')
            # Replace alpha-numeric with special token 'Client_Tok'
            if expanded_token.isalnum() and not expanded_token.isalpha() and not expanded_token.isdigit():
                expanded_token = 'ClientTok'
            if expanded_token.isdigit():
                expanded_token = 'NUM'
            query.extend(expanded_token.split())
        processed_x.append(query)
    return processed_x


def load_glove():
    embeddings_index = {}
    with open(os.path.join('data/', 'glove.6B.100d.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    return embeddings_index


def get_embedding_matrix(embedding_index, **kwargs):
    words_not_found = []
    vocab = len(kwargs['word_index']) + 1

    # 0.25 is chosen so the unknown vectors have same variance as pre-trained ones
    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(vocab, kwargs['embedding_dim']))
    for word in kwargs['word_index']:
        if kwargs['word_index'][word] >= vocab:
            continue
        embedding_vector = embedding_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[kwargs['word_index'][word]] = embedding_vector
        else:
            words_not_found.append(word)
    
    words_not_found = set(words_not_found)
    logging.info("{} words not found : {}".format(len(words_not_found), words_not_found))
    return embedding_matrix