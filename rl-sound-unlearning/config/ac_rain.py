"""Configuration for rain sound unlearning."""
from config.base import get_config as get_base_config


def get_config():
    """Configuration for unlearning rain sounds."""
    config = get_base_config()
    
    config["run_name"] = "ac_rain_unlearning"
    config["target_concept"] = "Rain"
    config["logdir"] = "logs/ac_rain"
    
    config["num_epochs"] = 30
    config["train"]["batch_size"] = 4

    config["prompt_fn"] = "rain_descriptions"
    
    return config
