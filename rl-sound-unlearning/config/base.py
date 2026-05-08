"""Base configuration for audio model unlearning."""


def get_config():
    """Returns base configuration dictionary for audio unlearning."""
    config = {
        # General
        "run_name": "",
        "seed": 42,
        "logdir": "logs",
        "num_epochs": 30,
        "save_freq": 10,
        "eval_freq": 5,
        "num_checkpoint_limit": 5,

        # Audio processing
        "sample_rate": 16000,
        "audio_length_seconds": 1.0,
        "num_samples": 16000,

        # Training
        "train": {
            "batch_size": 4,
            "critic_lr": 1e-3,
            "generator_lr": 3e-4,
            "critic_warmup_epochs": 20,
        },

        # REINFORCE reward weights
        "reward_weights": {
            "unlearn": 1.0,
            "realism": 0.4,
            "spec_entropy": 0.4,
            "anti_periodic": 0.5,
            "in_batch_div": 0.4,
            "retain_cls": 1.0,
            "retain_audio": 0.4,
        },
        "entropy_coef": 1e-4,
        
        # Model
        "model": {
            "critic_hidden_channels": 64,
            "critic_kernel_size": 5,
            "generator_latent_dim": 100,
            "generator_hidden_channels": 32,
            "generator_kernel_size": 5,
            "residual_scale": 0.6,
        },
        
        # Sampling
        "sample": {
            "batch_size": 4,
            "num_batches_per_epoch": 4,
        },
        
        # Reward
        "reward_fn": "concept_unlearning_reward",
        "target_concept": None,  # Will be set by concept-specific config
        
        # Data
        "data_path": "data",
        "concept_data_path": "data/concept_data",
        "description_path": "data/concept_descriptions",
    }
    
    return config
