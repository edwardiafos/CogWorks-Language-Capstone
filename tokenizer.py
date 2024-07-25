import string
from pathlib import Path
import json

def process_caption(caption):
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.lower()

    words = caption.split()

    return words
