"""Hugging Face Hub helper utilities.

Centralizes auth token resolution for gated models (e.g., Llama family)
so we don't rely on hardcoded secrets or per-call inconsistencies.

Never commit tokens to the repo. Use env vars instead.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolve a Hugging Face access token.

    Precedence:
    1) explicit_token (caller-provided)
    2) HF_TOKEN
    3) HUGGINGFACE_HUB_TOKEN
    4) HUGGINGFACE_TOKEN (legacy)

    Returns None if no token is available.
    """
    if explicit_token:
        return explicit_token

    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def resolve_local_model_path(model_id: str, hf_token: Optional[str] = None) -> str:
    """
    Resolve a HuggingFace model ID to its local cache path using snapshot_download.
    
    When in offline mode, we need to pass the actual local path instead of
    a Hub model ID to avoid API calls during model_info() checks.
    
    Args:
        model_id: HuggingFace model ID or local path
        hf_token: Optional HuggingFace token
        
    Returns:
        Local filesystem path to the model
    """
    from huggingface_hub import snapshot_download
    
    # If it's already a local path, return as-is
    if os.path.isdir(model_id):
        return model_id
    
    # Use snapshot_download with local_files_only=True to get cached path
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_files_only=True,
            token=hf_token,
        )
        return local_path
    except Exception as e:
        # If resolution fails, return original ID (let caller handle error)
        return model_id
