from typing import Dict, Any

import pyobs.object
from astroplan import Observer

from pyobs_cloudcover.world_model import WorldModel, SimpleModel
from pyobs_cloudcover.world_model.wcs_model_loader import WCSModelLoader


class WorldModelFactory(object):
    def __init__(self, observer: Observer):
        self._observer = observer

    def __call__(self, world_model_config: Dict[str, Any]) -> WorldModel:
        if world_model_config['class'] == 'pyobs_cloudcover.world_model.SimpleModel':
            model: SimpleModel = pyobs.object.get_object(world_model_config, SimpleModel)
            return model
        if world_model_config['class'] == 'pyobs_cloudcover.world_model.WCSModel':
            wcs_model_loader = WCSModelLoader(world_model_config["file_path"])
            return wcs_model_loader()

        raise ValueError(f'World model class {world_model_config["class"]} not supported')