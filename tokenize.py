import string

def process_caption(caption):
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.lower()

    words = caption.split()

    return words

bob = process_caption("Hello, my name is bob. I liked farming!!")
print(bob)