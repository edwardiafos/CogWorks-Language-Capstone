import string
from pathlib import Path
from cogworks_data.language import get_data_path

# Load stop words from file
def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip() for line in file)

stopwords_path = get_data_path("stopwords.txt")
stopwords = load_stopwords(stopwords_path)

def process_caption(caption: str, remove_stops: bool = False) -> list:

    #* Remove punctuation and convert to lowercase
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.lower()

    #* Split the caption into words
    words = caption.split()

    #* Optionally remove stop words
    if remove_stops:
        words = [word for word in words if word not in stopwords]

    return words
