import pandas as pd
from abc import ABC, abstractmethod
from jax import jacfwd

class Friend(ABC):

    def __init__(self,infile:str,
                 outfile:str,
                 keys_to_keep:list):
        
        self.infile = infile
        self.outfile = outfile
        self.keys_to_keep = keys_to_keep

        self._load_df()

        self.output = {}

    def _load_df(self):
        self.df = pd.read_parquet(self.config["dataframe"])

    def _save_output(self):
        df = pd.DataFrame(self.output)
        df.to_parquet(self.outfile)

    def add_key(self,key,values):
        self.output[key] == values

    @staticmethod
    @abstractmethod
    def calc_weights(input_params:dict):
        """
        Method that needs to be defined in child class

        Return: List of total weights for each event
        """
        pass
    
    def calc_grad_weights(self,input_params):
        grad_weights = jacfwd(self.calc_weights(input_params))
        return grad_weights
    
    def create_dataframe(self,input_params):
        self.output["weights"] = self.calc_weights(input_params)
        self.output["grad_weights"] = self.calc_grad_weights(input_params)

        for key in self.keys_to_keep:
            self.output[key] = self.df[key]
