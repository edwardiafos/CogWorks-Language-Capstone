import mygrad
import mynn
import numpy as np

from mygrad.nnet.initializers import glorot_normal
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
        self.dense = dense(input_size=512, output_size=d, weight_initializer=glorot_normal, bias=True)

    def __call__(self, x: np.ndarray):
        """x should be a (N, 512) array"""
        return self.dense(x)
    
    @property
    def parameters(self):
        return self.dense.parameters
