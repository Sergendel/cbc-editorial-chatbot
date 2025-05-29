import json
from pathlib import Path

from config.config import Config


def extract_metadata(metadata_files):
    """
    Extracts clearly defined metadata (ids, urls, publish times, timestamps)
    from provided metadata JSON files.
    """
    extracted_metadata = {
        "ids": set(),
        "urls": set(),
        "publish_times": set(),
        "last_update_times": set(),
        "timestamps": set(),  # for guidelines
    }

    for file_path in metadata_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                # IDs (news articles)
                item_id = item.get("id", "")
                if item_id:
                    extracted_metadata["ids"].add(item_id)

                # URLs (guidelines)
                url = item.get("url", "")
                if url:
                    extracted_metadata["urls"].add(url)

                # Publish times (news articles)
                publish_time = item.get("publish_time", "")
                if publish_time:
                    extracted_metadata["publish_times"].add(publish_time)

                # Last update times (news articles)
                last_update = item.get("last_update", "")
                if last_update:
                    extracted_metadata["last_update_times"].add(last_update)

                # Timestamps (guidelines)
                timestamp = item.get("timestamp", "")
                if timestamp:
                    extracted_metadata["timestamps"].add(timestamp)

    # Convert sets to sorted lists for clear serialization
    for key in extracted_metadata:
        extracted_metadata[key] = sorted(list(extracted_metadata[key]))

    return extracted_metadata


def save_metadata(metadata, save_path):
    """
    Saves the extracted metadata dictionary into a structured JSON file.
    Ensures that the directory exists before saving.
    """
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Config
    project_root = Path(__file__).parent.parent.parent.resolve()
    config_path = project_root / "config" / "config.yml"
    config = Config(str(config_path))

    # Absolute paths from the project root
    metadata_files = [
        project_root / config.news_metadata_path,
        project_root / config.guidelines_metadata_path,
    ]

    metadata = extract_metadata(metadata_files)

    # Path for saving extracted metadata lookup file
    save_metadata(metadata, project_root / config.metadata_lookup_path)

    print("✅ Metadata extraction completed and saved.")
