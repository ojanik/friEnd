import pandas as pd
from abc import ABC, abstractmethod
from jax import jacfwd
import jax.numpy as jnp
import numpy as onp
from pathlib import Path


class Friend(ABC):

    def __init__(
        self,
        dfs: dict,          # this is actually a dict, not a list
        model_params: dict,
        clear: bool = False,
    ):
        self.model_params = model_params

        # Build model functions (implemented in subclasses)
        self.models = self.get_models()

        # Input file paths dict: {dataset_name: path}
        self.dfs = dfs
        self.frozen_keys = list(self.dfs.keys())
        print("Frozen keys:", self.frozen_keys)

        # Load dataframes and optionally clear old grad_* columns
        for k in self.frozen_keys:
            df = self._load_df(self.dfs[k])
            if clear:
                # Drop any previously created grad_weights_* columns
                cols_to_drop = [col for col in df.columns if "grad_weights" in col]
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
            self.dfs[k + "_df"] = df

        # Print true lengths of the loaded dataframes
        for k in self.frozen_keys:
            print(f"{k}: {len(self.dfs[k + '_df'])} rows")

    @staticmethod
    def _load_df(filename, **kwargs):
        ext = Path(filename).suffix.lower()

        if ext in (".csv", ".txt"):
            return pd.read_csv(filename, **kwargs)
        elif ext in (".h5", ".hdf", ".hdf5"):
            return pd.read_hdf(filename, **kwargs)
        elif ext in (".parquet", ".pq"):
            return pd.read_parquet(filename, **kwargs)
        elif ext in (".feather", ".ft"):
            return pd.read_feather(filename, **kwargs)
        elif ext in (".pkl", ".pickle"):
            return pd.read_pickle(filename, **kwargs)
        elif ext in (".json",):
            return pd.read_json(filename, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _save_output(self):
        for k in self.frozen_keys:
            outpath = Path(self.dfs[k])
            outpath = outpath.with_suffix(".parquet")
            self.dfs[k + "_df"].to_parquet(outpath)
            print(f"--- Saved dataframe to {outpath}")

    @abstractmethod
    def get_models(self):
        """Return a dict mapping dataset key -> callable(params) -> weights"""
        pass

    def calc_weights(self, model, input_params: dict, key: str):
        # model is a dict of callables, indexed by dataset key
        weights = model[key](input_params)
        return weights

    def calc_grad_weights(self, model, input_params: dict, key: str):
        grad_weights = jacfwd(model[key])(input_params)
        return grad_weights

    def add_weights(self):

        for k in self.frozen_keys:
            input_params = self.model_params
            print(f"Getting flux for {k}")

            # weights: 1D array with length == number of events in this dataset
            weights = self.calc_weights(self.models, input_params, k)

            # Check length match for safety (will raise a clearer error, if any)
            n_rows = len(self.dfs[k + "_df"])
            if weights.shape[0] != n_rows:
                raise ValueError(
                    f"Length mismatch for '{k}': weights ({weights.shape[0]}) "
                    f"!= dataframe rows ({n_rows})"
                )

            self.dfs[k + "_df"]["weights"] = onp.array(weights)

            grad_weights = self.calc_grad_weights(self.models, input_params, k)

            # grad_weights is a dict: {param_name: jnp.ndarray}
            for param_key in grad_weights.keys():
                arr = grad_weights[param_key]

                # Drop totally uninformative parameters
                if jnp.sum(jnp.abs(arr)) < 1e-30:
                    print(f"Dropped {param_key} from {k}.")
                    continue

                add_grad_weights = onp.array(arr)
                add_grad_weights[onp.isnan(add_grad_weights)] = 0.0

                # Again, verify length
                if add_grad_weights.shape[0] != n_rows:
                    raise ValueError(
                        f"Grad length mismatch for '{k}', param '{param_key}': "
                        f"{add_grad_weights.shape[0]} != {n_rows}"
                    )

                self.dfs[k + "_df"][f"grad_weights_{param_key}"] = add_grad_weights