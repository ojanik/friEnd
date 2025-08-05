import pandas as pd
from abc import ABC, abstractmethod
from jax import jacfwd
import jax.numpy as jnp

class Friend(ABC):

    def __init__(self,infile:str,
                 outfile:str,
                 keys_to_keep:list):
        
        self.infile = infile
        self.outfile = outfile
        self.keys_to_keep = keys_to_keep

        self.model = self.get_model()

        self._load_df()

        self.output = {}

    def _load_df(self):
        self.df = pd.read_parquet(self.infile)

    def _save_output(self):
        df = pd.DataFrame(self.output)
        df.to_parquet(self.outfile)

    def add_key(self,key,values):
        self.output[key] == values

    @abstractmethod
    def get_model(self):
        pass

    def calc_weights(self,input_params:dict):
        weights = self.model(input_params)
        return weights
    
    def calc_grad_weights(self,input_params):
        grad_weights = jacfwd(self.model)(input_params)
        return grad_weights
    
    def create_dataframe(self,input_params):
        self.output["weights"] = self.calc_weights(input_params)
        grad_weights = self.calc_grad_weights(input_params)

        for key in grad_weights.keys():
            self.output["grad_weights_"+key] = grad_weights[key]

        for key in self.keys_to_keep:
            self.output[key] = jnp.array(self.df[key])
