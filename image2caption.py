import mygrad
import mynn
import numpy as np

from mygrad.nnet.initializers import he_normal
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam

class Image2Caption:
    def __init__(self, d: int=200):
        """
        Parameters:

        d: int
            the embedding dimension we want to convert the image descriptor vector to so that 
            with the input being (N, 512), the output of our model is shape (N, d)
        """
        self.d = d
        self.dense = dense(input_size=512, output_size=d, weight_initializer=he_normal, bias=True)

    def __call__(self, x):
        return self.dense(x)
    
    @property
    def parameters(self):
        return self.dense.parameters