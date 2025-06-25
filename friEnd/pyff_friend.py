import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from copy import deepcopy

from functools import partial

import pyForwardFolding as pyFF

from .friend import Friend

class PyFF_Friend(Friend):
    def __init__(self,df_path,outfile,input_vars,pyffconfig):
        super().__init__(df_path,outfile)
        self.pyffconfig = pyffconfig
        self.input_variables = input_vars

    def get_models(self):
        model = pyFF.config.models_from_config(model = pyFF.config.models_from_config(self.pyffconfig))
        model = partial(model.evaluate,input_variables=self.input_variables)
        return model
