# Audio Model Unlearning - CV-Style Structure

This directory contains the audio model unlearning implementation, structured to match the CV (visual) version in `rl-machine-unlearning-ac-concept-removal`.

## Directory Structure

```
rl-sound-unlearning/
├── config/                          # Configuration files
│   ├── __init__.py
│   ├── base.py                      # Base configuration
│   ├── ac_rain.py                   # Rain concept unlearning config
│   ├── ac_wind.py                   # Wind concept unlearning config
│   └── ac_thunder.py                # Thunder concept unlearning config
├── data/
│   ├── concept_data/                # Audio data organized by concept
│   │   ├── Rain/
│   │   ├── Wind/
│   │   └── Thunder/
│   ├── concept_descriptions/        # Text descriptions of concepts
│   │   ├── rain_descriptions.txt
│   │   ├── wind_descriptions.txt
│   │   └── thunder_descriptions.txt
│   └── manifest.json                # Dataset metadata
├── audio_prompts.py                 # Description loader (like prompts.py)
├── audio_critic.py                  # Audio concept classifier
├── audio_generator.py               # Audio waveform generator
├── audio_dataset.py                 # PyTorch dataset class
├── audio_rewards.py                 # Reward computation
├── audio_utils.py                   # Utility functions
├── train_audio_unlearning.py        # Main training script
└── scripts/
    └── run_audio_unlearning.sh      # Training launcher script
```

## Usage

### Option 1: Using Config Files (Recommended)
```bash
# Train with rain concept config
python3 train_audio_unlearning.py --config ac_rain --save-samples

# Train with wind concept config
python3 train_audio_unlearning.py --config ac_wind --save-samples

# Train with thunder concept config
python3 train_audio_unlearning.py --config ac_thunder --save-samples
```

### Option 2: Legacy Command-Line Arguments (Backward Compatible)
```bash
# Train with manual arguments
python3 train_audio_unlearning.py \
  --data-dir data \
  --target-concept Rain \
  --epochs 20 \
  --batch-size 2 \
  --lr 3e-4 \
  --save-samples
```

### Option 3: Using Bash Script
```bash
bash scripts/run_audio_unlearning.sh
```

## Configuration System

### Base Configuration (`config/base.py`)
Contains default hyperparameters:
- Audio processing: sample rate, audio length, number of samples
- Training: learning rates, batch sizes, number of epochs
- Model architecture: hidden channels, kernel sizes
- Sampling and reward parameters

### Concept-Specific Configs (`config/ac_*.py`)
Each concept has its own config that inherits from `base.py` and can override:
- `target_concept`: The concept to unlearn (Rain, Wind, Thunder)
- `run_name`: Name for logging and checkpoints
- `logdir`: Output directory for this concept
- Training hyperparameters specific to the concept

### Adding New Concepts
1. Create a new config file: `config/ac_myconcept.py`
   ```python
   from config.base import get_config as get_base_config
   
   def get_config():
       config = get_base_config()
       config["target_concept"] = "MyNewConcept"
       config["run_name"] = "ac_myconcept_unlearning"
       # Override other parameters as needed
       return config
   ```

2. Create description file: `data/concept_descriptions/myconcept_descriptions.txt`
   - One description per line
   - Example: "Sound description here"

3. Add loader function to `audio_prompts.py`:
   ```python
   def myconcept_descriptions():
       return from_file("data/concept_descriptions/myconcept_descriptions.txt")
   ```

4. Organize training data in `data/concept_data/MyNewConcept/`

## Data Format

### Manifest Format (`data/manifest.json`)
```json
[
  {"audio": "path/to/audio.wav", "concept": "Rain"},
  {"audio": "path/to/audio.wav", "concept": "Wind"},
  {"audio": "path/to/audio.wav", "concept": "Thunder"}
]
```

### Audio Specifications
- Format: WAV (16-bit PCM)
- Sample Rate: 16000 Hz
- Duration: 1.0 second (16000 samples)
- Channels: Mono

## Concept Descriptions Format

Each file in `data/concept_descriptions/` contains multiple descriptions:
```
Heavy rain falling on roof
Steady rainfall with distant thunder
Rain pattering on windows
Light drizzle in the forest
...
```

The training system randomly samples descriptions from these files during training.

## Training Process

1. **Critic Training**: Learns to classify audio by concept
2. **Generator Training**: Learns to generate audio that *reduces* target concept probability
3. **Reward Computation**: Uses critic predictions to compute reinforcement learning rewards

## Output

Training creates checkpoints in `logs/ac_<concept>/` with:
- Model weights: `audio_unlearning_epoch_*.pt`
- Generated samples: `sample_epoch_*.wav` (if `--save-samples` is used)
- Training logs printed to console

## Similarity to CV Version

This audio implementation mirrors the CV version architecture:
- **Configs**: Base + concept-specific (like CV's `base.py` + `ac_Cats.py`)
- **Descriptions**: Text files per concept (like CV's `sd_prompt_*.txt`)
- **Prompts**: Loader functions (like CV's `prompts.py`)
- **Training**: Actor-Critic with concept unlearning rewards
- **Data**: Organized by concept with manifests
