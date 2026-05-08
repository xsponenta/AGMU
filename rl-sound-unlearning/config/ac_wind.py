"""Configuration for wind sound unlearning."""
from config.base import get_config as get_base_config


def get_config():
    """Configuration for unlearning wind sounds."""
    config = get_base_config()
    
    config["run_name"] = "ac_wind_unlearning"
    config["target_concept"] = "Wind"
    config["logdir"] = "logs/ac_wind"
    
    config["num_epochs"] = 30
    config["train"]["batch_size"] = 4

    config["prompt_fn"] = "wind_descriptions"
    
    return config
