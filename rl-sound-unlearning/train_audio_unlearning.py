"""REINFORCE training loop for audio concept unlearning via edit-policy.

Pipeline per epoch:
  1. Train critic one step on a real-audio batch (cross-entropy).
  2. Sample a real-audio batch, run policy stochastically, get edited audio.
  3. Compute multi-term reward (unlearn + realism + anti-collapse).
  4. REINFORCE update on the policy with a moving-average baseline.

Output sample is the *deterministic mean* edit applied to a real input clip,
so it is always a real track -- not silence/noise.
"""
import argparse
import csv
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audio_dataset import AudioConceptDataset
from audio_generator import AudioEditPolicy
from audio_critic import AudioCritic
from audio_rewards import RewardWeights, compute_rewards
from audio_utils import save_audio


def load_config(name: str):
    if name.endswith(".py"):
        name = name[:-3]
    try:
        module = importlib.import_module(f"config.{name}")
    except ImportError:
        module = importlib.import_module(name)
    return module.get_config()


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE audio unlearning via edit-policy.")
    p.add_argument("--config", type=str, help="Config module (e.g. ac_rain)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--target-concept", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Legacy shortcut for generator learning rate")
    p.add_argument("--critic-lr", type=float, default=None)
    p.add_argument("--generator-lr", type=float, default=None)
    p.add_argument("--critic-warmup-epochs", type=int, default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--save-samples", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(critic, policy, loader, device, num_concepts, target_idx, sample_rate):
    critic.eval()
    if policy is not None:
        policy.eval()

    total = 0
    clean_correct = 0
    edited_correct = 0
    clean_non_target_correct = 0
    edited_non_target_correct = 0
    clean_target_prob = 0.0
    edited_target_prob = 0.0
    changed_rms = 0.0
    target_total = 0
    clean_target_only_prob = 0.0
    edited_target_only_prob = 0.0
    non_target_total = 0

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        batch = waveforms.size(0)
        target_mask = labels == target_idx
        non_target_mask = ~target_mask

        clean_probs = F.softmax(critic(waveforms), dim=-1)
        clean_pred = clean_probs.argmax(dim=-1)
        clean_correct += (clean_pred == labels).sum().item()
        if non_target_mask.any():
            clean_non_target_correct += (clean_pred[non_target_mask] == labels[non_target_mask]).sum().item()
            non_target_total += non_target_mask.sum().item()
        clean_target_prob += clean_probs[:, target_idx].sum().item()
        if target_mask.any():
            clean_target_only_prob += clean_probs[target_mask, target_idx].sum().item()
            target_total += target_mask.sum().item()

        if policy is not None:
            cond = torch.zeros(batch, num_concepts, device=device)
            cond[:, target_idx] = 1.0
            edited = policy.act_mean(waveforms, cond)
            edited_probs = F.softmax(critic(edited), dim=-1)
            edited_pred = edited_probs.argmax(dim=-1)
            edited_correct += (edited_pred == labels).sum().item()
            if non_target_mask.any():
                edited_non_target_correct += (
                    edited_pred[non_target_mask] == labels[non_target_mask]
                ).sum().item()
            edited_target_prob += edited_probs[:, target_idx].sum().item()
            if target_mask.any():
                edited_target_only_prob += edited_probs[target_mask, target_idx].sum().item()
            changed_rms += (edited - waveforms).pow(2).mean(dim=(1, 2)).sqrt().sum().item()

        total += batch

    metrics = {
        "critic_acc": clean_correct / max(total, 1),
        "clean_non_target_acc": clean_non_target_correct / max(non_target_total, 1),
        "clean_target_prob": clean_target_prob / max(total, 1),
        "clean_target_only_prob": clean_target_only_prob / max(target_total, 1),
    }
    if policy is not None:
        metrics.update({
            "edited_acc": edited_correct / max(total, 1),
            "edited_non_target_acc": edited_non_target_correct / max(non_target_total, 1),
            "edited_target_prob": edited_target_prob / max(total, 1),
            "edited_target_only_prob": edited_target_only_prob / max(target_total, 1),
            "edit_rms": changed_rms / max(total, 1),
        })
    return metrics


def train_critic_epoch(critic, loader, optimizer, loss_fn, device):
    critic.train()
    total_loss = 0.0
    steps = 0
    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = critic(waveforms)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "data_path": "data",
            "target_concept": "Rain",
            "num_epochs": 30,
            "train": {"batch_size": 4, "critic_lr": 1e-3, "generator_lr": 3e-4},
            "logdir": "checkpoints",
            "sample_rate": 16000,
        }

    # CLI overrides
    if args.data_dir: config["data_path"] = args.data_dir
    if args.target_concept: config["target_concept"] = args.target_concept
    if args.epochs is not None: config["num_epochs"] = args.epochs
    if args.batch_size is not None: config["train"]["batch_size"] = args.batch_size
    if args.lr is not None: config["train"]["generator_lr"] = args.lr
    if args.critic_lr is not None: config["train"]["critic_lr"] = args.critic_lr
    if args.generator_lr is not None: config["train"]["generator_lr"] = args.generator_lr
    if args.critic_warmup_epochs is not None:
        config["train"]["critic_warmup_epochs"] = args.critic_warmup_epochs

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    torch.manual_seed(seed)

    checkpoint_dir = Path(args.checkpoint_dir or config.get("logdir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = config.get("sample_rate", 16000)

    dataset = AudioConceptDataset(
        config["data_path"],
        sample_rate=sample_rate,
        duration=config.get("audio_length_seconds", 1.0),
    )
    if len(dataset) == 0:
        raise ValueError(f"Empty dataset at {config['data_path']}. Run data_synth.py first.")
    if config["target_concept"] not in dataset.concept_to_idx:
        raise ValueError(
            f"Target '{config['target_concept']}' not in dataset concepts: {dataset.concepts}"
        )

    batch_size = config["train"]["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    num_concepts = len(dataset.concepts)
    target_idx = dataset.concept_to_idx[config["target_concept"]]

    model_config = config.get("model", {})
    critic = AudioCritic(
        num_concepts=num_concepts,
        hidden_channels=model_config.get("critic_hidden_channels", 64),
    ).to(device)
    policy = AudioEditPolicy(
        num_concepts=num_concepts,
        hidden=model_config.get("generator_hidden_channels", 32),
        residual_scale=model_config.get("residual_scale", 0.6),
    ).to(device)

    critic_lr = config["train"].get("critic_lr", 1e-3)
    policy_lr = config["train"].get("generator_lr", 3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    ce = nn.CrossEntropyLoss()

    weights = RewardWeights(
        unlearn=config.get("reward_weights", {}).get("unlearn", 1.0),
        realism=config.get("reward_weights", {}).get("realism", 0.4),
        spec_entropy=config.get("reward_weights", {}).get("spec_entropy", 0.4),
        anti_periodic=config.get("reward_weights", {}).get("anti_periodic", 0.5),
        in_batch_div=config.get("reward_weights", {}).get("in_batch_div", 0.4),
        retain_cls=config.get("reward_weights", {}).get("retain_cls", 1.0),
        retain_audio=config.get("reward_weights", {}).get("retain_audio", 0.4),
    )

    # REINFORCE moving-average baseline
    baseline = torch.zeros(1, device=device)
    baseline_momentum = 0.9
    entropy_coef = config.get("entropy_coef", 1e-4)

    num_epochs = config["num_epochs"]
    save_freq = config.get("save_freq", 10)
    eval_freq = config.get("eval_freq", save_freq)

    # Pick a fixed real input for sample audios so we can compare across epochs.
    fixed_audio, fixed_label = dataset[0]
    for i in range(len(dataset)):
        wave, lab = dataset[i]
        if lab == target_idx:
            fixed_audio = wave
            break
    fixed_audio = fixed_audio.unsqueeze(0).to(device)
    save_audio(fixed_audio[0], checkpoint_dir / "input_reference.wav", sample_rate=sample_rate)
    metrics_path = checkpoint_dir / "metrics.csv"
    train_metrics_path = checkpoint_dir / "train_metrics.csv"
    metric_fieldnames = [
        "epoch",
        "critic_acc",
        "clean_non_target_acc",
        "edited_non_target_acc",
        "clean_target_prob",
        "clean_target_only_prob",
        "edited_target_prob",
        "edited_target_only_prob",
        "edit_rms",
    ]
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames, extrasaction="ignore")
        writer.writeheader()

    train_metric_fieldnames = [
        "epoch",
        "critic_loss",
        "policy_loss",
        "reward",
        "baseline",
        "unlearn_reward",
        "realism_reward",
        "spec_entropy_reward",
        "anti_periodic_reward",
        "diversity_reward",
        "retain_cls_reward",
        "retain_audio_reward",
        *metric_fieldnames[1:],
    ]
    with open(train_metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=train_metric_fieldnames, extrasaction="ignore")
        writer.writeheader()

    metrics = evaluate(critic, None, loader, device, num_concepts, target_idx, sample_rate)
    print(
        "Initial critic | "
        f"acc={metrics['critic_acc']:.3f} "
        f"target_p={metrics['clean_target_prob']:.3f} "
        f"target_only_p={metrics['clean_target_only_prob']:.3f}"
    )

    critic_warmup_epochs = config["train"].get("critic_warmup_epochs", 0)
    for warm_epoch in range(1, critic_warmup_epochs + 1):
        loss = train_critic_epoch(critic, loader, critic_opt, ce, device)
        if warm_epoch == 1 or warm_epoch % max(1, critic_warmup_epochs // 5) == 0:
            metrics = evaluate(critic, None, loader, device, num_concepts, target_idx, sample_rate)
            print(
                f"Critic warmup {warm_epoch:03d}/{critic_warmup_epochs} | "
                f"loss={loss:.3f} acc={metrics['critic_acc']:.3f} "
                f"target_only_p={metrics['clean_target_only_prob']:.3f}"
            )

    for epoch in range(1, num_epochs + 1):
        critic.train()
        policy.train()

        ep_critic = 0.0
        ep_policy = 0.0
        ep_components = {k: 0.0 for k in ["unlearn", "realism", "spec_entropy",
                                          "anti_periodic", "diversity",
                                          "retain_cls", "retain_audio"]}
        ep_total = 0.0
        steps = 0

        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            B = waveforms.size(0)

            # --- Critic step (supervised) ---
            critic_opt.zero_grad()
            logits = critic(waveforms)
            loss_critic = ce(logits, labels)
            loss_critic.backward()
            critic_opt.step()
            ep_critic += loss_critic.item()

            # --- Policy step (REINFORCE) ---
            # Condition on the *target* concept onehot (we want to unlearn it
            # from any input audio, including non-target ones).
            cond = torch.zeros(B, num_concepts, device=device)
            cond[:, target_idx] = 1.0

            for p in critic.parameters():
                p.requires_grad_(False)
            critic.eval()

            edited, log_prob_sum, mu, log_sigma = policy(waveforms, cond)
            rewards, components = compute_rewards(
                critic, edited, target_idx, weights, sample_rate=sample_rate,
                original=waveforms, labels=labels,
            )

            # Update baseline (EMA over batch means)
            batch_mean = rewards.mean().detach()
            baseline.mul_(baseline_momentum).add_(batch_mean * (1.0 - baseline_momentum))
            advantage = rewards - baseline

            # REINFORCE: maximize E[advantage * log_prob] => minimize -mean(...)
            # log_prob_sum is per-sample summed log-prob; normalize by length so
            # the gradient magnitude stays comparable across audio durations.
            T = edited.size(-1)
            policy_loss = -(advantage * log_prob_sum / T).mean()

            # Entropy bonus on the Gaussian policy: H = sum(log_sigma) + const.
            entropy = log_sigma.mean()
            policy_loss = policy_loss - entropy_coef * entropy

            policy_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            policy_opt.step()

            for p in critic.parameters():
                p.requires_grad_(True)

            ep_policy += policy_loss.item()
            ep_total += rewards.mean().item()
            for k, v in components.items():
                ep_components[k] += v.mean().item()
            steps += 1

        for k in ep_components:
            ep_components[k] /= max(steps, 1)
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"critic={ep_critic/steps:.3f} policy={ep_policy/steps:.3f} "
            f"R={ep_total/steps:.3f} baseline={baseline.item():.3f} | "
            f"unlearn={ep_components['unlearn']:.2f} "
            f"real={ep_components['realism']:.2f} "
            f"ent={ep_components['spec_entropy']:.2f} "
            f"aper={ep_components['anti_periodic']:.2f} "
            f"div={ep_components['diversity']:.2f} "
            f"retain={ep_components['retain_cls']:.2f}/{ep_components['retain_audio']:.2f}"
        )

        metrics = evaluate(critic, policy, loader, device, num_concepts, target_idx, sample_rate)
        if epoch % eval_freq == 0 or epoch == num_epochs:
            print(
                f"Eval {epoch:03d} | "
                f"critic_acc={metrics['critic_acc']:.3f} "
                f"non_target_acc={metrics['edited_non_target_acc']:.3f} "
                f"clean_target_p={metrics['clean_target_prob']:.3f} "
                f"target_clean_p={metrics['clean_target_only_prob']:.3f} "
                f"target_edited_p={metrics['edited_target_only_prob']:.3f} "
                f"edited_target_p={metrics['edited_target_prob']:.3f} "
                f"edit_rms={metrics['edit_rms']:.3f}"
            )
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                row = {"epoch": epoch, **metrics}
                writer = csv.DictWriter(f, fieldnames=metric_fieldnames, extrasaction="ignore")
                writer.writerow(row)

        with open(train_metrics_path, "a", newline="", encoding="utf-8") as f:
            row = {
                "epoch": epoch,
                "critic_loss": ep_critic / max(steps, 1),
                "policy_loss": ep_policy / max(steps, 1),
                "reward": ep_total / max(steps, 1),
                "baseline": baseline.item(),
                "unlearn_reward": ep_components["unlearn"],
                "realism_reward": ep_components["realism"],
                "spec_entropy_reward": ep_components["spec_entropy"],
                "anti_periodic_reward": ep_components["anti_periodic"],
                "diversity_reward": ep_components["diversity"],
                "retain_cls_reward": ep_components["retain_cls"],
                "retain_audio_reward": ep_components["retain_audio"],
                **metrics,
            }
            writer = csv.DictWriter(f, fieldnames=train_metric_fieldnames, extrasaction="ignore")
            writer.writerow(row)

        if args.save_samples:
            policy.eval()
            with torch.no_grad():
                cond = torch.zeros(1, num_concepts, device=device)
                cond[:, target_idx] = 1.0
                edited = policy.act_mean(fixed_audio, cond)
                save_audio(edited[0], checkpoint_dir / f"sample_epoch_{epoch}.wav",
                           sample_rate=sample_rate)

        if epoch % save_freq == 0 or epoch == num_epochs:
            torch.save({
                "epoch": epoch,
                "critic_state": critic.state_dict(),
                "policy_state": policy.state_dict(),
                "concepts": dataset.concepts,
                "target_concept": config["target_concept"],
            }, checkpoint_dir / f"audio_unlearning_epoch_{epoch}.pt")

    print(f"Done. Checkpoints + samples in {checkpoint_dir}")


if __name__ == "__main__":
    train()
