"""Reproducible Phase-A training, evaluation, and replay CLI for FlyTaiko."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Required by CUDA for deterministic cuBLAS execution when seeded training is
# requested. It must be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

from flyconnectome.loader import generate_synthetic_connectome
from flyconnectome.lif_network import LIFNetwork
from taiko.environment import ACTION_NAMES, NUM_ACTIONS, TaikoEnvironment


RAW_FEATURE_SIZE = 16


class Note:
    def __init__(self, time_ms: float, note_type: str):
        self.time_ms = float(time_ms)
        self.note_type = note_type

    @property
    def is_hit(self):
        return True

    @property
    def is_don(self):
        return self.note_type in ("don", "don_big")

    @property
    def is_big(self):
        return self.note_type in ("don_big", "kat_big")


class Metadata:
    def __init__(self, raw: dict):
        self.title = raw.get("title", "Unknown")
        self.artist = raw.get("artist", "Unknown")
        self.version = raw.get("version", raw.get("difficulty", "Unknown"))
        self.overall_difficulty = float(raw.get("overall_difficulty", 5.0))


class JsonBeatmap:
    def __init__(self, raw: dict):
        self.metadata = Metadata(raw.get("metadata", {}))
        self.notes = sorted(
            [Note(n.get("t", n.get("time_ms")), n.get("type", n.get("note_type")))
             for n in raw.get("notes", [])],
            key=lambda n: n.time_ms,
        )


class Encoder:
    def __init__(self, n_sensory: int, device: torch.device, lookahead: int = 8):
        self.n = n_sensory
        self.device = device
        self.lookahead = lookahead
        self.npn = n_sensory // (lookahead + 2)

    def encode(self, state: dict) -> torch.Tensor:
        out = torch.zeros(self.n, device=self.device)
        for i, (dt, note_type) in enumerate(state["upcoming_notes"][: self.lookahead]):
            if note_type == "none":
                continue
            start = i * self.npn
            q = self.npn // 4
            urgency = 1 / (1 + max(dt, 0) / 150) if dt > 0 else max(0, 1 + dt / 200)
            hit_now = max(0, 1 - abs(dt) / 15)
            is_don = 1.0 if "don" in note_type else -1.0
            out[start:start + q] = urgency * 15
            out[start + q:start + 2 * q] = hit_now * 25
            out[start + 2 * q:start + 3 * q] = is_don * urgency * 10
            out[start + 3 * q:start + 4 * q] = -is_don * urgency * 10
        combo_start = self.lookahead * self.npn
        out[combo_start:combo_start + self.npn] = min(state.get("combo", 0) / 50, 1) * 5
        return out


def raw_features(state: dict, device: torch.device) -> torch.Tensor:
    out = torch.zeros(RAW_FEATURE_SIZE, device=device)
    for i, (dt, note_type) in enumerate(state["upcoming_notes"][:3]):
        if note_type == "none":
            continue
        j = i * 4
        out[j] = 1 / (1 + max(dt, 0) / 150)
        out[j + 1] = max(0, 1 - abs(dt) / 15)
        out[j + 2] = 1 if "don" in note_type else -1
        out[j + 3] = 1 if "big" in note_type else 0
    out[12] = float(state.get("don_alt", False))
    out[13] = float(state.get("kat_alt", False))
    return out


class Decoder(nn.Module):
    def __init__(self, lif_size: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lif_size + RAW_FEATURE_SIZE, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class Config:
    seed: int = 42
    neurons: int = 10000
    lif_steps: int = 30
    step_ms: int = 16
    hidden: int = 256
    cooldown_ms: int = 24


class Runtime:
    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.conn = generate_synthetic_connectome(cfg.neurons, seed=cfg.seed)
        self.lif = LIFNetwork(self.conn.adjacency, device=device)
        self.encoder = Encoder(len(self.conn.sensory_ids), self.device)
        self.decoder = Decoder(len(self.conn.readout_ids) * 2, cfg.hidden).to(self.device)

    def reset(self):
        self.lif.reset()

    @torch.no_grad()
    def state_vector(self, state: dict):
        current = self.lif.inject_current(self.conn.sensory_ids, self.encoder.encode(state))
        for _ in range(self.cfg.lif_steps):
            self.lif.step(current)
        return torch.cat((self.lif.get_state_vector(self.conn.readout_ids),
                          raw_features(state, self.device)))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_maps(path: Path, max_maps: int | None = None) -> list[JsonBeatmap]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = raw if isinstance(raw, list) else [raw]
    maps = [JsonBeatmap(row) for row in rows if row.get("notes")]
    return maps[:max_maps] if max_maps else maps


def split_by_song(maps: list[JsonBeatmap], val_fraction: float, seed: int):
    titles = sorted({(m.metadata.artist, m.metadata.title) for m in maps})
    random.Random(seed).shuffle(titles)
    n_val = max(1, round(len(titles) * val_fraction)) if len(titles) > 1 else 0
    val_titles = set(titles[:n_val])
    return ([m for m in maps if (m.metadata.artist, m.metadata.title) not in val_titles],
            [m for m in maps if (m.metadata.artist, m.metadata.title) in val_titles])


def iter_frames(runtime: Runtime, beatmap: JsonBeatmap, max_frames: int | None = None):
    env = TaikoEnvironment(beatmap, step_ms=runtime.cfg.step_ms)
    runtime.reset()
    frames = 0
    while not env.is_done and (max_frames is None or frames < max_frames):
        state = env.get_state()
        expert = env.get_expert_action()  # Same state as features, before step.
        yield env, state, runtime.state_vector(state), expert
        env.step(expert)
        frames += 1


def checkpoint(runtime: Runtime, cfg: Config, epoch: int, optimizer, best_metric: float):
    return {
        "phase": "A", "epoch": epoch, "seed": cfg.seed,
        "num_actions": NUM_ACTIONS, "action_names": ACTION_NAMES,
        "layout": {"don": ["F", "J"], "kat": ["D", "K"],
                   "don_big": "F+J", "kat_big": "D+K"},
        "config": asdict(cfg), "decoder": runtime.decoder.state_dict(),
        "optimizer": optimizer.state_dict(), "best_metric": best_metric,
    }


def train(runtime: Runtime, train_maps: list[JsonBeatmap], val_maps: list[JsonBeatmap],
          epochs: int, lr: float, output: Path, resume: Path | None,
          max_frames: int | None, patience: int = 5):
    optimizer = torch.optim.Adam(runtime.decoder.parameters(), lr=lr)
    weights = torch.tensor([0.05, 1, 1, 1, 1, 2, 2], device=runtime.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    start, best = 0, -math.inf
    if resume:
        saved = torch.load(resume, map_location=runtime.device, weights_only=False)
        runtime.decoder.load_state_dict(saved["decoder"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved.get("epoch", 0)), float(saved.get("best_metric", -math.inf))
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "config.json").write_text(json.dumps(asdict(runtime.cfg), indent=2) + "\n")
    log_path = output.parent / "train.jsonl"
    stale = 0
    for epoch in range(start, epochs):
        runtime.decoder.train()
        losses = []
        for map_index, beatmap in enumerate(train_maps):
            states, labels = [], []
            cache_path = output.parent / "features" / f"{map_index:04d}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location=runtime.device, weights_only=True)
                states = list(cached["states"].unbind())
                labels = cached["labels"]
            else:
                print(json.dumps({"event": "collect", "map": map_index + 1,
                                  "total_maps": len(train_maps), "title": beatmap.metadata.title}), flush=True)
                for _, _, vector, expert in iter_frames(runtime, beatmap, max_frames):
                    states.append(vector)
                    labels.append(expert)
                    if len(states) % 1000 == 0:
                        print(json.dumps({"event": "collect_progress", "map": map_index + 1,
                                          "frames": len(states), "expert_hits": sum(x > 0 for x in labels)}), flush=True)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                if states:
                    torch.save({"states": torch.stack(states).cpu(), "labels": labels}, cache_path)
            if not states:
                continue
            labels_t = torch.tensor(labels, device=runtime.device)
            hits = torch.nonzero(labels_t > 0).flatten()
            none = torch.nonzero(labels_t == 0).flatten()
            if not len(hits):
                continue
            keep_none = none[torch.randperm(len(none), device=runtime.device)[: min(2 * len(hits), len(none))]]
            indices = torch.cat((hits, keep_none))
            indices = indices[torch.randperm(len(indices), device=runtime.device)]
            states_t = torch.stack(states)
            for offset in range(0, len(indices), 64):
                idx = indices[offset:offset + 64]
                loss = loss_fn(runtime.decoder(states_t[idx]), labels_t[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; stopping immediately")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            print(json.dumps({"event": "train_map", "epoch": epoch + 1,
                              "map": map_index + 1, "loss": float(np.mean(losses))}), flush=True)
            # Recovery snapshot during long first-epoch feature extraction.
            torch.save(checkpoint(runtime, runtime.cfg, epoch, optimizer, best),
                       output.parent / "last.pt")
        runtime.decoder.eval()
        metrics = evaluate(runtime, val_maps or train_maps[:1], max_frames=max_frames)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else None,
               "validation": metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        torch.save(checkpoint(runtime, runtime.cfg, epoch + 1, optimizer, best),
                   output.parent / "last.pt")
        score = metrics["great_rate"] - metrics["miss_rate"]
        if score > best:
            best = score
            stale = 0
            torch.save(checkpoint(runtime, runtime.cfg, epoch + 1, optimizer, best), output)
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        reason = None
        if epoch + 1 >= 3 and metrics["miss_rate"] > .90:
            reason = "validation miss rate exceeds 90% after warmup"
        elif stale >= patience:
            reason = "validation plateau"
        elif epoch + 1 >= 3 and score < best - .15:
            reason = "validation regression"
        if reason:
            (output.parent / "stop.json").write_text(json.dumps({"reason": reason, "epoch": epoch + 1}))
            print(json.dumps({"event": "early_stop", "reason": reason}), flush=True)
            break


def _density(notes: list[Note]) -> float:
    if len(notes) < 2:
        return 0.0
    return 1000 * (len(notes) - 1) / max(notes[-1].time_ms - notes[0].time_ms, 1)


@torch.no_grad()
def evaluate(runtime: Runtime, maps: Iterable[JsonBeatmap], max_frames: int | None = None):
    runtime.decoder.eval()
    counts = {"great": 0, "good": 0, "miss": 0}
    errors, signed, type_hits, type_total = [], [], {}, {}
    alt_violations = 0
    map_rows = []
    for beatmap in maps:
        env = TaikoEnvironment(beatmap, step_ms=runtime.cfg.step_ms)
        runtime.reset(); last_hit = -9999; frames = 0
        previous = {"don": None, "kat": None}
        while not env.is_done and (max_frames is None or frames < max_frames):
            state = env.get_state()
            vector = runtime.state_vector(state)
            action = int(runtime.decoder(vector).argmax())
            if action and state["current_time_ms"] - last_hit < runtime.cfg.cooldown_ms:
                action = 0
            if action:
                last_hit = state["current_time_ms"]
            target = env.hit_notes[env.next_note_idx] if env.next_note_idx < len(env.hit_notes) else None
            result = env.step(action)
            if result.judgment in counts:
                note_type = target.note_type if target else result.note_type
                type_total[note_type] = type_total.get(note_type, 0) + 1
                if result.judgment in ("great", "good"):
                    type_hits[note_type] = type_hits.get(note_type, 0) + 1
                    err = state["current_time_ms"] - target.time_ms
                    signed.append(err); errors.append(abs(err))
                    color = "don" if "don" in note_type else "kat"
                    if action in (1, 2, 3, 4):
                        hand = action in (2, 4)
                        alt_violations += int(previous[color] == hand)
                        previous[color] = hand
            frames += 1
        summary = env.get_results_summary()
        counts["great"] += summary["greats"]
        counts["good"] += summary["goods"]
        counts["miss"] += summary["misses"]
        map_rows.append({"title": beatmap.metadata.title, "density_notes_s": _density(beatmap.notes),
                         **summary})
    total = sum(counts.values()) or 1
    abs_arr = np.asarray(errors or [math.nan])
    signed_arr = np.asarray(signed or [math.nan])
    return {
        **counts, "great_rate": counts["great"] / total, "good_rate": counts["good"] / total,
        "miss_rate": counts["miss"] / total,
        "mean_abs_error_ms": float(np.nanmean(abs_arr)), "median_abs_error_ms": float(np.nanmedian(abs_arr)),
        "p90_abs_error_ms": float(np.nanpercentile(abs_arr, 90)), "p95_abs_error_ms": float(np.nanpercentile(abs_arr, 95)),
        "mean_signed_error_ms": float(np.nanmean(signed_arr)),
        "early": int(np.sum(signed_arr < 0)), "late": int(np.sum(signed_arr > 0)),
        "full_alt_violations": alt_violations,
        "accuracy_by_type": {k: type_hits.get(k, 0) / v for k, v in type_total.items()},
        "maps": map_rows,
    }


def load_model(runtime: Runtime, path: Path):
    saved = torch.load(path, map_location=runtime.device, weights_only=False)
    if saved.get("num_actions") != NUM_ACTIONS:
        raise ValueError("Checkpoint does not declare the canonical 7-action space")
    runtime.decoder.load_state_dict(saved["decoder"])


@torch.no_grad()
def write_replay(runtime: Runtime, beatmap: JsonBeatmap, output: Path,
                 max_frames: int | None = None):
    runtime.decoder.eval(); runtime.reset()
    env = TaikoEnvironment(beatmap, step_ms=runtime.cfg.step_ms)
    actions, trace = [], []; last_hit = -9999; frames = 0
    keys = ["NONE", "F", "J", "D", "K", "F+J", "D+K"]
    while not env.is_done and (max_frames is None or frames < max_frames):
        state = env.get_state(); vector = runtime.state_vector(state)
        logits = runtime.decoder(vector); action = int(logits.argmax())
        if action and state["current_time_ms"] - last_hit < runtime.cfg.cooldown_ms:
            action = 0
        if action:
            last_hit = state["current_time_ms"]
        target = env.hit_notes[env.next_note_idx] if env.next_note_idx < len(env.hit_notes) else None
        result = env.step(action)
        if action:
            target_time = target.time_ms if target and result.judgment in ("great", "good") else state["current_time_ms"]
            actions.append({"time_ms": int(target_time), "model_time_ms": int(state["current_time_ms"]),
                            "action": keys[action], "judgment": result.judgment})
        trace.append({"time_ms": int(state["current_time_ms"]),
                      "mean_rate": runtime.lif.get_stats()["mean_firing_rate"],
                      "active_neurons": runtime.lif.get_stats()["active_neurons"],
                      "logits": logits.detach().cpu().tolist()})
        frames += 1
        if frames % 1000 == 0:
            print(json.dumps({"event": "replay_progress", "frames": frames,
                              "time_ms": state["current_time_ms"],
                              "notes_processed": env.next_note_idx,
                              "total_notes": len(env.hit_notes)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 2,
                                  "metadata": {"title": beatmap.metadata.title},
                                  "actions": actions, "neural_trace": trace}, indent=2) + "\n")
    return env.get_results_summary()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("train", "evaluate", "replay"))
    p.add_argument("--data", type=Path, default=Path("target_map.json"))
    p.add_argument("--output", type=Path, default=Path("runs/phase_a/best.pt"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neurons", type=int, default=10000)
    p.add_argument("--lif-steps", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-maps", type=int)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args(); seed_all(args.seed)
    cfg = Config(seed=args.seed, neurons=args.neurons, lif_steps=args.lif_steps)
    maps = load_maps(args.data, args.max_maps)
    train_maps, val_maps = split_by_song(maps, args.val_fraction, args.seed)
    runtime = Runtime(cfg, args.device)
    print(json.dumps({"device": str(runtime.device), "train_maps": len(train_maps),
                      "validation_maps": len(val_maps), "config": asdict(cfg)}))
    if args.command == "train":
        train(runtime, train_maps or maps, val_maps, args.epochs, args.lr, args.output,
              args.resume, args.max_frames)
    elif args.command == "evaluate":
        if not args.checkpoint:
            raise SystemExit("evaluate requires --checkpoint")
        load_model(runtime, args.checkpoint)
        print(json.dumps(evaluate(runtime, val_maps or maps, args.max_frames), indent=2))
    else:
        if not args.checkpoint:
            raise SystemExit("replay requires --checkpoint")
        load_model(runtime, args.checkpoint)
        print(json.dumps(write_replay(runtime, maps[0], args.output, args.max_frames), indent=2))


if __name__ == "__main__":
    main()
