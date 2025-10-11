"""FastAPI server for running inference with a pretrained HRM checkpoint."""

import argparse
import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import _load_config, _load_metadata, _prepare_batch
from pretrain import PretrainConfig, init_train_state


def _select_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


class InferenceService:
    """Keep a model loaded in memory and serve inference requests."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        default_return_keys: Optional[Iterable[str]] = None,
    ) -> None:
        if default_return_keys is None:
            default_return_keys = ("logits", "q_halt_logits", "q_continue_logits")

        self._default_return_keys = tuple(default_return_keys)
        self._torch_device = _select_device(device)

        self._config: PretrainConfig = _load_config(checkpoint_path)
        self._metadata = _load_metadata(self._config.data_path)

        self._config.eval_save_outputs = list(self._default_return_keys)
        self._config.checkpoint_path = os.path.dirname(checkpoint_path)

        self._train_state = init_train_state(
            self._config,
            self._metadata,
            world_size=1,
            device=self._torch_device,
        )

        state_dict = torch.load(checkpoint_path, map_location=self._torch_device)
        try:
            self._train_state.model.load_state_dict(state_dict, assign=True)
        except Exception:
            self._train_state.model.load_state_dict(
                {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()},
                assign=True,
            )

        self._train_state.model.to(self._torch_device)
        self._train_state.model.eval()

    def predict(
        self,
        input_array: np.ndarray,
        puzzle_identifier: int = 0,
        return_keys: Optional[Iterable[str]] = None,
    ) -> dict:
        if return_keys is None:
            return_keys = self._default_return_keys

        batch = _prepare_batch(
            input_array,
            self._metadata,
            puzzle_identifier,
            self._torch_device,
        )

        outputs = {}
        with torch.inference_mode():
            with torch.device(self._torch_device):
                carry = self._train_state.model.initial_carry(batch)

            while True:
                carry, _loss, _metrics, preds, all_finish = self._train_state.model(
                    carry=carry,
                    batch=batch,
                    return_keys=return_keys,
                )
                outputs.update(preds)

                if bool(all_finish):
                    break

        def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
            tensor = tensor.detach()
            if tensor.dtype in (torch.float16, torch.bfloat16):
                tensor = tensor.to(torch.float32)
            return tensor.cpu().numpy()

        return {key: _to_numpy(value) for key, value in outputs.items()}


class InferenceRequest(BaseModel):
    input_array: Union[List[int], List[List[int]]]
    puzzle_identifier: int = 0
    return_keys: Optional[List[str]] = None
    only_predictions: bool = True


class InferenceResponse(BaseModel):
    predictions: Optional[List[int]]
    outputs: Dict[str, Dict[str, Any]] | None = None


def create_app(service: InferenceService) -> FastAPI:
    app = FastAPI(title="HRM Inference Server")

    @app.post("/infer", response_model=InferenceResponse)
    async def infer(request: InferenceRequest) -> InferenceResponse:
        try:
            input_array = np.asarray(request.input_array, dtype=np.int32)
        except ValueError as exc:  # pragma: no cover - numpy raises ValueError
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            outputs = await asyncio.to_thread(
                service.predict,
                input_array,
                request.puzzle_identifier,
                request.return_keys,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        formatted_outputs = {
            key: {"shape": list(value.shape), "values": value.tolist()}
            for key, value in outputs.items()
        }

        predictions: Optional[List[int]] = None
        if "logits" in outputs:
            predicted_tokens = np.argmax(outputs["logits"][0], axis=-1)
            predictions = predicted_tokens.tolist()

        if request.only_predictions:
            return InferenceResponse(outputs=None, predictions=predictions)
        return InferenceResponse(outputs=formatted_outputs, predictions=predictions)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a FastAPI server for HRM inference.",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the model checkpoint."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to.")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run inference on (default: auto-detect).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    service = InferenceService(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    app = create_app(service)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
