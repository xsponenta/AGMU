"""Config for SpeechT5 word-unlearning via rejection-sampling DPO."""


def get_config():
    return {
        # IO
        "pairs_dir": "dpo_pairs/run01",
        "out_dir": "logs/tts_dpo",

        # Optimization
        "num_epochs": 4,
        "batch_size": 1,
        "lr": 3e-5,
        "beta": 0.05,               # DPO temperature (lower => smaller drift)
        "sft_coef": 0.2,            # SFT weight on chosen mels for target pairs
        "retain_sft_coef": 1.0,     # SFT weight on retain-anchor mels (pin retain)

        # LoRA
        "lora_r": 4,                # smaller adapter => less retain damage

        "seed": 0,
    }
