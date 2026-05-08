"""List of audio concepts available for unlearning."""

# Available audio concepts for unlearning
CONCEPTS = [
    "Rain",
    "Wind", 
    "Thunder",
]

# Concept to config module mapping
CONCEPT_CONFIGS = {
    "Rain": "config.ac_rain",
    "Wind": "config.ac_wind",
    "Thunder": "config.ac_thunder",
}

# Concept descriptions
CONCEPT_DESCRIPTIONS = {
    "Rain": "Rainfall, precipitation, water droplets",
    "Wind": "Wind sounds, gusts, breeze noise",
    "Thunder": "Thunderclaps, rumbles, storm noise",
}
