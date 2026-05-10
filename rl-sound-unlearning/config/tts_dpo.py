"""Config for SpeechT5 word-unlearning via rejection-sampling DPO."""


def get_config():
    return {
        # IO
        "pairs_dir": "dpo_pairs/run01",
        "out_dir": "logs/tts_dpo",

        # Optimization
        "num_epochs": 15,
        "batch_size": 1,
        "lr": 1e-4,
        "beta": 1.0,                # DPO temperature; mel L1 loss is O(1) so beta must be O(1)
        "sft_coef": 0.4,            # SFT weight on chosen mels for target pairs
        "retain_sft_coef": 0.3,     # don't let retain SFT drown the DPO signal

        # LoRA
        "lora_r": 16,

        "seed": 0,
    }
