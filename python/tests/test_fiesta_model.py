import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp


MODEL_PATH = Path(__file__).parents[1] / "survey_sim" / "fiesta_model.py"


class FakeFluxModel:
    parameter_names = ["p"]

    def __init__(self, name, filters=None):
        self.filters = filters

    def predict(self, params):
        return jnp.array([0.2, 1.0]), {
            "lsstg": jnp.array([10.0, 11.0]),
        }

    def vpredict(self, params):
        return jnp.array([[0.2, 1.0]]), {
            "lsstg": jnp.array([[10.0, 11.0]]),
        }


def load_model():
    fiesta = types.ModuleType("fiesta")
    inference = types.ModuleType("fiesta.inference")
    lightcurve_model = types.ModuleType("fiesta.inference.lightcurve_model")
    lightcurve_model.FluxModel = FakeFluxModel
    modules = {
        "fiesta": fiesta,
        "fiesta.inference": inference,
        "fiesta.inference.lightcurve_model": lightcurve_model,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("test_fiesta_model_impl", MODEL_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.FiestaKNModel(filters=["lsstg"])


class FiestaModelTests(unittest.TestCase):
    def test_predict_masks_pre_explosion_observations(self):
        model = load_model()
        _, mags = model.predict(
            {
                "p": 1.0,
                "luminosity_distance": 40.0,
                "redshift": 0.0,
                "_t_exp": 100.0,
                "_obs_times_mjd": [99.0, 100.2, 101.0],
                "_obs_bands": ["g", "g", "g"],
            }
        )

        self.assertEqual(mags["g"], [99.0, 10.0, 11.0])

    def test_batch_predict_masks_pre_explosion_observations(self):
        model = load_model()
        results = model.batch_predict(
            [
                {
                    "p": 1.0,
                    "luminosity_distance": 40.0,
                    "redshift": 0.0,
                    "_t_exp": 100.0,
                    "_obs_times_mjd": [99.0, 100.2, 101.0],
                    "_obs_bands": ["g", "g", "g"],
                }
            ]
        )

        self.assertEqual(results[0][1]["g"], [99.0, 10.0, 11.0])


if __name__ == "__main__":
    unittest.main()