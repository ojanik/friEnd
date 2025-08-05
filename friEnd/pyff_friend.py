import jax
jax.config.update("jax_enable_x64", True)

from functools import partial

import pyForwardFolding as pyFF

from .friend import Friend

class PyFF_Friend(Friend):
    def __init__(self,df_path,outfile,keys_to_keep,
                 input_vars,pyffconfig,dataset_name):
        self.pyffconfig = pyffconfig
        self.input_variables = input_vars
        self.dataset_name = dataset_name
        super().__init__(df_path,outfile,keys_to_keep)
        

    def get_model(self):
        model = pyFF.config.models_from_config(self.pyffconfig)[self.dataset_name]
        def model_partial(input_params):
            return model.evaluate(self.input_variables,input_params)
        return model_partial
