"""Configuration for thunder sound unlearning."""
from config.base import get_config as get_base_config


def get_config():
    """Configuration for unlearning thunder sounds."""
    config = get_base_config()
    
    config["run_name"] = "ac_thunder_unlearning"
    config["target_concept"] = "Thunder"
    config["logdir"] = "logs/ac_thunder"
    
    config["num_epochs"] = 30
    config["train"]["batch_size"] = 4

    config["prompt_fn"] = "thunder_descriptions"
    
    return config
