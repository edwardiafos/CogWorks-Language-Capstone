import string
from pathlib import Path
import json

with open(get_data_path("stopwords.txt"), 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

def process_caption(caption, remove_stops=False):
    
    # optionally remove stop-words, default at False for speed purposes
    
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.lower()

    words = caption.split()
    words = [w for w in word if w in string.ascii_lowercase]
    
    if remove_stops:
        words = [w for w in words if w not in stops]

    return words
