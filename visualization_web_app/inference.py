import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import json
import config
import earth2mip.initial_conditions

from earth2mip import registry, inference_ensemble
from earth2mip.initial_conditions import cds
from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load
from earth2mip.schema import EnsembleRun
from earth2mip.inference_ensemble import run_inference
from earth2mip.schema import Grid, PerturbationStrategy

def main():
    # Set number of GPUs to use to 1
    os.environ["WORLD_SIZE"] = "1"
    # Set model registry as a local folder
    model_registry = os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")
    os.makedirs(model_registry, exist_ok=True)
    os.environ["MODEL_REGISTRY"] = model_registry
    print(f"{os.environ['MODEL_REGISTRY']}")

    print("loading FCNv2 small model, this can take a bit")
    config = config.CONFIG
    config_str = json.dumps(config)
    inference_ensemble.main(config_str)

if __name__ == "__main__":
    main()
    