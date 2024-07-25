import numpy as np
from pathlib import Path
import json
import pickle
from cogworks_data.language import get_data_path
from typing import List, Union, Sequence
from collections import Counter
from pathlib import Path
import os
from cogworks_data.language import get_data_path
from pathlib import Path
import json
from tokenizer import process_caption

from gensim.models import KeyedVectors
from coco_data import COCODataManager

filename = "glove.6B.200d.txt.w2v"
### this takes a while to load -- keep this in mind when designing your capstone project ###
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

### Load saved image descriptor vectors ###
resnet18_features = {}
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)


def initialize_coco_data():
    """
    coco_data can be used using the following:
    
    coco_data = initialize_coco_data()
    image_ids = coco_data.get_image_ids()
    captions = coco_data.get_caption_ids()
    image_to_captions = coco_data.image_to_caption()
    caption_id_to_images = coco_data.caption_id_to_image()
    caption_id_to_captions = coco_data.caption_id_to_caption()
    """
    filename = get_data_path("captions_train2014.json")
    coco_data = COCODataManager(filename)
    return coco_data

def get_user_input():
    pass



def se_text(text: str, captions: Sequence[str]) -> np.ndarray: # um someone who has taken more math than algebra II please check this lol
    """Takes text and returns a shape (50,) array by using IDF and glove embeddings.
        Returns a shape (N, 50) array with N being the number of tokens in `text.`
    """
    global glove 
    text_tokens = process_caption(text) # len N
    caption_tokens = [process_caption(cap) for cap in captions]

    total_tokens = [token for cap in captions for token in process_caption(cap)] + text_tokens

    vocab = set(total_tokens)
    counters = []
    for caption_token in caption_tokens:
        counters.append(Counter(caption_token))
    counters.append(Counter(text_tokens))

    N = len(counters)
    nt = [sum(1 if t in counter else 0 for counter in counters) for t in text_tokens]
    nt = np.array(nt, dtype=float)
    idf = np.log10(N / nt) # shape (N,)

    glove_embeddings = np.array([glove[word] for word in text_tokens]) # shape (N, 50)
    for i, weight in enumerate(idf):
        glove_embeddings[i] += weight

    ret = glove_embeddings / np.linalg.norm(glove_embeddings) # shape (N, 50)
    return ret.mean(axis=0) # should be shape (50,)? hopefully??



def se_image(): # this is Image2Caption we need to create/train
    pass


