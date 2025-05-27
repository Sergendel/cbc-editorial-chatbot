import json
from pathlib import Path

import numpy as np


def load_embeddings(embeddings_path, metadata_path):
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(embeddings) == len(
        metadata
    ), f"Embeddings ({len(embeddings)}) vs metadata ({len(metadata)}) mismatch!"
    print(f"Loaded explicitly {len(embeddings)} embeddings.")
    return embeddings, metadata


# Explicitly define base path relative to this file
BASE_DIR = (
    Path(__file__).resolve().parent.parent
)  # adjust if your file is deeper in structure

if __name__ == "__main__":
    embeddings, metadata = load_embeddings(
        BASE_DIR / "data/embeddings_storage/guidelines_embeddings.npy",
        BASE_DIR / "data/embeddings_storage/guidelines_metadata.json",
    )

    print(f"Embeddings shape: {embeddings.shape}")
