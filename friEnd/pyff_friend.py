import jax
jax.config.update("jax_enable_x64", True)

from functools import partial

import pyForwardFolding as pyFF

from .friend import Friend


class PyFF_Friend(Friend):
    def __init__(
        self,
        pyffconfig,
        model_params,
        clear: bool = False,
    ):
        self.pyffconfig = pyffconfig

        # Load inputs and config from pyForwardFolding
        inputs = pyFF.config.dataset_from_config(pyffconfig)
        config = pyFF.config._load_config(pyffconfig)
        self.config = config

        # dfs: dataset_name -> path
        dfs = {x["name"]: x["path"] for x in config["datasets"]}
        print("dfs:", dfs)

        # Store inputs as dict: dataset_name -> input_dict
        self.inputs = {x["name"]: inputs[x["name"]] for x in config["datasets"]}

        for name, inp in self.inputs.items():
            print(f"{name}: {len(inp['log10_reco_energy'])} events")

        # Call base class constructor
        super().__init__(dfs, model_params, clear=clear)

    def get_models(self):
        """
        Build a mapping:
            models[dataset_name] = lambda params: m.evaluate(dataset_inputs, params)
        so that for each dataset we have a callable that takes only the params
        and returns per-event weights.
        """
        pyff_models = pyFF.config.models_from_config(self.pyffconfig)

        # Build a mapping histogram_name -> { model_name -> dataset_name }
        model_map = {}
        for h in self.config["histograms"]:
            hkey = h["name"]         # e.g. "tracks", "cascades", "doubles"
            data_models = h["models"]  # e.g. [["neutrino_model", "tracks_selection"]]
            model_map[hkey] = {}
            for model_name, dataset_name in data_models:
                model_map[hkey][model_name] = dataset_name

        print("model map:", model_map)

        models = {}

        # pyff_models is keyed by histogram name, e.g. "tracks", "cascades", "doubles"
        for h in pyff_models.keys():
            model_dict = pyff_models[h]      # dict: {model_name: model_object}
            print("MODEL KEYS:", model_dict.keys())

            for model_name, m in model_dict.items():
                print(f"Building model for {model_name}")

                dataset_name = model_map[h][model_name]
                print(f"Corresponding dataset: {dataset_name}")

                input_data = self.inputs[dataset_name]
                print(f"{dataset_name} has {len(input_data['log10_reco_energy'])} events")

                # m.evaluate is assumed to be: evaluate(input_data, params)
                models[dataset_name] = partial(m.evaluate, input_data)

        print("Models", models.keys())

        # Now Friend.add_weights expects self.models to be indexed by dataset key,
        # which matches the keys in dfs (tracks_selection, cascade_selection, double_selection)
        return models