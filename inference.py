"""Utility script to run inference with a pretrained HRM checkpoint."""
import argparse
import json
import os
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import yaml

from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID
from pretrain import PretrainConfig, init_train_state


def _load_config(checkpoint_path: str) -> PretrainConfig:
    config_path = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find all_config.yaml next to checkpoint at {checkpoint_path}"
        )

    with open(config_path, "r") as f:
        return PretrainConfig(**yaml.safe_load(f))


def _load_metadata(data_path: str) -> PuzzleDatasetMetadata:
    metadata_path = os.path.join(data_path, "train", "dataset.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Expected dataset metadata at "
            f"{metadata_path}. Please make sure the dataset is available."
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return PuzzleDatasetMetadata(**metadata)


def _prepare_batch(
    input_array: np.ndarray,
    metadata: PuzzleDatasetMetadata,
    puzzle_identifier: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if input_array.ndim == 1:
        input_array = np.expand_dims(input_array, axis=0)

    if input_array.ndim != 2:
        raise ValueError(
            "Input array must have shape (seq_len,) or (batch, seq_len). "
            f"Got shape {tuple(input_array.shape)}."
        )

    batch_size, seq_len = input_array.shape
    if seq_len > metadata.seq_len:
        raise ValueError(
            "Input sequence length exceeds model maximum of "
            f"{metadata.seq_len}. Got {seq_len}."
        )

    if seq_len < metadata.seq_len:
        pad_width = metadata.seq_len - seq_len
        input_array = np.pad(
            input_array,
            ((0, 0), (0, pad_width)),
            constant_values=metadata.pad_id,
        )

    inputs = input_array.astype(np.int32, copy=False)
    labels = np.full_like(inputs, IGNORE_LABEL_ID, dtype=np.int32)

    if not 0 <= puzzle_identifier < metadata.num_puzzle_identifiers:
        raise ValueError(
            "Puzzle identifier must be within [0, num_puzzle_identifiers). "
            f"Got {puzzle_identifier} with maximum "
            f"{metadata.num_puzzle_identifiers - 1}."
        )

    puzzle_identifiers = np.full(
        (batch_size,), puzzle_identifier, dtype=np.int32
    )

    batch_np = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
    }

    return {k: torch.from_numpy(v).to(device) for k, v in batch_np.items()}


def run_inference(
    checkpoint: str,
    input_array: np.ndarray,
    puzzle_identifier: int = 0,
    return_keys: Optional[Iterable[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load a checkpoint and run inference on ``input_array``.

    Parameters
    ----------
    checkpoint:
        Path to the checkpoint file produced by training.
    input_array:
        Numpy array containing token ids of shape ``(seq_len,)`` or
        ``(batch, seq_len)``.
    puzzle_identifier:
        Identifier of the puzzle to use for the puzzle embedding. Must be in the
        valid range defined by the dataset metadata.
    return_keys:
        Iterable of model outputs to return. Defaults to
        ``("logits", "q_halt_logits", "q_continue_logits")``.
    device:
        Torch device string. Defaults to ``"cuda"`` if available, then ``"mps"``
        on Apple Silicon, otherwise ``"cpu"``.
    """

    if return_keys is None:
        return_keys = ("logits", "q_halt_logits", "q_continue_logits")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)

    config = _load_config(checkpoint)
    metadata = _load_metadata(config.data_path)

    config.eval_save_outputs = list(return_keys)
    config.checkpoint_path = os.path.dirname(checkpoint)

    train_state = init_train_state(
        config, metadata, world_size=1, device=torch_device
    )

    state_dict = torch.load(checkpoint, map_location=device)
    try:
        train_state.model.load_state_dict(state_dict, assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()},
            assign=True,
        )

    train_state.model.eval()

    batch = _prepare_batch(input_array, metadata, puzzle_identifier, torch_device)

    with torch.inference_mode():
        carry = train_state.model.initial_carry(batch)
        outputs: Dict[str, torch.Tensor] = {}

        while True:
            carry, _loss, _metrics, preds, all_finish = train_state.model(
                carry=carry, batch=batch, return_keys=return_keys
            )
            outputs.update(preds)

            if bool(all_finish):
                break

    return {k: v.cpu().numpy() for k, v in outputs.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a HRM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint.")
    parser.add_argument(
        "--input-array",
        required=True,
        help="Path to a .npy file containing the input array.",
    )
    parser.add_argument(
        "--puzzle-identifier",
        type=int,
        default=0,
        help="Puzzle identifier to use for the puzzle embedding.",
    )
    parser.add_argument(
        "--return-keys",
        nargs="*",
        default=("logits", "q_halt_logits", "q_continue_logits"),
        help="Model outputs to return.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run inference on (default: auto-detect).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    input_array = np.load(args.input_array)

    outputs = run_inference(
        checkpoint=args.checkpoint,
        input_array=input_array,
        puzzle_identifier=args.puzzle_identifier,
        return_keys=args.return_keys,
        device=args.device,
    )

    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}\n{value}")


if __name__ == "__main__":
    main()
