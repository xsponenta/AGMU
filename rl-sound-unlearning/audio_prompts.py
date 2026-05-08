"""Audio prompt/description loader for concept unlearning."""
import functools
import random
import os


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly.
    
    Args:
        path: Path to the description file
        
    Returns:
        List of description lines (stripped of whitespace)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def from_file(path, low=None, high=None):
    """
    Load descriptions from file and return a random one.
    
    Args:
        path: Path to the description file
        low: Optional lower index bound
        high: Optional upper index bound
        
    Returns:
        Tuple of (random_description, empty_dict)
    """
    descriptions = _load_lines(path)[low:high]
    if not descriptions:
        raise ValueError(f"No descriptions found in {path}")
    return random.choice(descriptions), {}


def rain_descriptions():
    """Load random rain sound description."""
    return from_file("data/concept_descriptions/rain_descriptions.txt")


def wind_descriptions():
    """Load random wind sound description."""
    return from_file("data/concept_descriptions/wind_descriptions.txt")


def thunder_descriptions():
    """Load random thunder sound description."""
    return from_file("data/concept_descriptions/thunder_descriptions.txt")
