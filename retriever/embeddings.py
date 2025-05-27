import json
from pathlib import Path
import numpy as np

def load_embeddings(embeddings_path, metadata_path):
    embeddings = np.load(embeddings_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert len(embeddings) == len(metadata), (
        f"Embeddings ({len(embeddings)}) vs metadata ({len(metadata)}) mismatch!"
    )
    print(f"Loaded explicitly {len(embeddings)} embeddings.")

    # Explicitly extract chunk texts from metadata
    texts = [item["chunk_text"] for item in metadata]

    return texts, embeddings, metadata

BASE_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    texts, embeddings, metadata = load_embeddings(
        BASE_DIR / "data/embeddings_storage/guidelines_embeddings.npy",
        BASE_DIR / "data/embeddings_storage/guidelines_metadata.json",
    )

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample text: {texts[0]}...")
