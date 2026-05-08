# Audio Unlearning - CV Version Structure Mapping

This document shows how the audio unlearning project mirrors the visual (CV) model unlearning structure.

## Structure Comparison

### Configuration System

| CV Version | Audio Version | Purpose |
|-----------|--------------|---------|
| `ddpo_pytorch/config/base.py` | `config/base.py` | Base configuration with defaults |
| `ddpo_pytorch/config/ac_Cats.py` | `config/ac_rain.py` | Concept-specific config (example: Cats → Rain) |
| `ddpo_pytorch/config/ac_Dogs.py` | `config/ac_wind.py` | Another concept config (example: Dogs → Wind) |
| — | `config/ac_thunder.py` | Third concept config (Thunder) |

### Data & Descriptions

| CV Version | Audio Version | Purpose |
|-----------|--------------|---------|
| `ddpo_pytorch/assets/20_concepts_removal/sd_prompt_Cats.txt` | `data/concept_descriptions/rain_descriptions.txt` | Descriptions for concept (example: Cats → Rain) |
| `ddpo_pytorch/assets/20_concepts_removal/sd_prompt_Dogs.txt` | `data/concept_descriptions/wind_descriptions.txt` | Another concept descriptions |
| — | `data/concept_descriptions/thunder_descriptions.txt` | Thunder concept descriptions |

### Prompt/Description Loaders

| CV Version | Audio Version | Purpose |
|-----------|--------------|---------|
| `ddpo_pytorch/prompts.py` | `audio_prompts.py` | Load descriptions from files |
| `prompts.from_file()` | `audio_prompts.from_file()` | Load and return random description |
| `prompts.cats_clip_dataset()` | `audio_prompts.rain_descriptions()` | Concept-specific loader (example: cats → rain) |
| `prompts.dogs_clip_dataset()` | `audio_prompts.wind_descriptions()` | Another concept loader |
| — | `audio_prompts.thunder_descriptions()` | Thunder loader |

### Concepts Lists

| CV Version | Audio Version | Purpose |
|-----------|--------------|---------|
| `concepts_list.py` | `audio_concepts_list.py` | Define available concepts |

### Training Scripts

| CV Version | Audio Version | Purpose |
|-----------|--------------|---------|
| `scripts/training/train.py` | `train_audio_unlearning.py` | Main training entry point |
| Uses config via arguments | Supports `--config ac_rain` | Config-based training |
| — | Supports legacy arguments | Backward compatible CLI |

### Data Organization

| CV Version | Audio Version | Purpose |
|-----------|--------------|---------|
| `assets/` | `data/` | Main data directory |
| `assets/20_concepts_removal/` | `data/concept_data/` | Organized by concept |
| `assets/*.txt` (individual files) | `data/concept_data/Rain/`, `Wind/`, `Thunder/` | Concept subdirectories |
| — | `data/manifest.json` | Dataset metadata |

## Usage Patterns

### CV Version Training Example
```bash
# Training visual model for cats unlearning
python scripts/training/train.py --config ac_Cats
```

### Audio Version Training Example
```bash
# Training audio model for rain unlearning
python3 train_audio_unlearning.py --config ac_rain
```

Both follow the same pattern: `--config ac_<concept>`

## Architecture Parallels

### Model Components

| CV | Audio | Purpose |
|----|-------|---------|
| Image Critic (CLIP classifier) | Audio Critic (1D CNN classifier) | Learn concept features |
| Image Generator (Diffusion) | Audio Generator (ConvTranspose) | Generate concept-free samples |
| Aesthetic Reward | Audio Reward | Concept unlearning objective |

### Training Loop

Both use Actor-Critic reinforcement learning:
1. **Critic Phase**: Learn to classify concept presence
2. **Generator Phase**: Generate samples that minimize concept probability
3. **Reward Computation**: Use critic to compute RL rewards

## Adding New Concepts

### CV Version
1. Create `config/ac_NewConcept.py`
2. Create `assets/20_concepts_removal/sd_prompt_NewConcept.txt`
3. Add loader in `prompts.py`: `def newconcept_clip_dataset()`

### Audio Version
1. Create `config/ac_newconcept.py`
2. Create `data/concept_descriptions/newconcept_descriptions.txt`
3. Add loader in `audio_prompts.py`: `def newconcept_descriptions()`

## File Organization Summary

```
CV Version (Visual)          Audio Version (Audio)
─────────────────           ────────────────────

ddpo_pytorch/
├── config/
│   ├── base.py         ←→  config/base.py
│   ├── ac_Cats.py      ←→  config/ac_rain.py
│   └── ...
├── assets/
│   └── 20_concepts_removal/
│       ├── sd_prompt_*.txt ←→ data/concept_descriptions/*.txt
│       └── ...
└── prompts.py          ←→  audio_prompts.py

train.py                ←→  train_audio_unlearning.py

concepts_list.py        ←→  audio_concepts_list.py
```

## Key Configuration Options

Both versions support concept-specific configs:

```python
# Base config has defaults
config["num_epochs"] = 101
config["train"]["batch_size"] = 2

# Concept config can override
config["target_concept"] = "Rain"
config["run_name"] = "ac_rain_unlearning"
```

## Benefits of This Structure

1. **Consistency**: Same architecture as proven CV version
2. **Scalability**: Easy to add new concepts
3. **Maintainability**: Centralized configs and descriptions
4. **Modularity**: Separate configs, prompts, data per concept
5. **Reproducibility**: Config-based training ensures consistent runs
6. **Documentation**: Clear mapping between concepts and files

## Testing the Structure

```bash
# Config-based training (recommended)
python3 train_audio_unlearning.py --config ac_rain

# Check concepts list
python3 -c "from audio_concepts_list import CONCEPTS; print(CONCEPTS)"

# Load descriptions
python3 -c "from audio_prompts import rain_descriptions; print(rain_descriptions())"
```
