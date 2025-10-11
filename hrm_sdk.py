"""Simple Python SDK for interacting with the HRM FastAPI inference server."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import requests


@dataclass
class InferenceResult:
    """Structured response from the HRM inference endpoint."""

    outputs: Dict[str, np.ndarray]
    predictions: Optional[List[int]]

    @classmethod
    def from_response(cls, data: Dict[str, Dict[str, Any]]) -> "InferenceResult":
        raw_outputs = data.get("outputs")
        parsed_outputs: Dict[str, np.ndarray] = {}

        if raw_outputs is not None:
            for key, payload in raw_outputs.items():
                values = payload.get("values", []) if isinstance(payload, dict) else []
                parsed_outputs[key] = np.asarray(values)

        predictions = data.get("predictions")
        if predictions is not None and not isinstance(predictions, list):
            raise TypeError("predictions must be a list or null")

        return cls(outputs=parsed_outputs, predictions=predictions)


class HRMInferenceClient:
    """Convenience wrapper for calling the HRM inference server from Python."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._own_session = session is None
        self._session = session or requests.Session()

    def close(self) -> None:
        if self._own_session:
            self._session.close()

    def __enter__(self) -> "HRMInferenceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def infer(
        self,
        input_array: Union[np.ndarray, Iterable[Iterable[int]], Iterable[int]],
        *,
        puzzle_identifier: int = 0,
        return_keys: Optional[Iterable[str]] = None,
    ) -> InferenceResult:
        payload = {
            "input_array": _ensure_serialisable_array(input_array),
            "puzzle_identifier": puzzle_identifier,
        }

        if return_keys is not None:
            payload["return_keys"] = list(return_keys)

        response = self._session.post(
            f"{self._base_url}/infer",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()

        return InferenceResult.from_response(response.json())


@contextlib.contextmanager
def hrm_client(
    base_url: str,
    *,
    timeout: float = 10.0,
    session: Optional[requests.Session] = None,
):
    """Context manager helper for the :class:`HRMInferenceClient`."""

    client = HRMInferenceClient(base_url, timeout=timeout, session=session)
    try:
        yield client
    finally:
        client.close()


def _ensure_serialisable_array(
    array_like: Union[np.ndarray, Iterable[Iterable[int]], Iterable[int]],
) -> List[List[int]]:
    """Convert array-like input to a nested list of integers."""

    array = np.asarray(array_like, dtype=np.int32)

    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    elif array.ndim != 2:
        raise ValueError("input_array must be 1D or 2D")

    return array.tolist()


__all__ = ["HRMInferenceClient", "InferenceResult", "hrm_client"]
