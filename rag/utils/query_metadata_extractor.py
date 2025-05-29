import re


def extract_metadata_from_query(query: str):
    """
    Extracts structured metadata (IDs, ISO timestamps, URLs)
    from user input queries using regex.
    """
    metadata = {}

    # Article ID extraction (e.g., "1.6272172")
    id_match = re.search(r"\b\d{1,2}\.\d{6,8}\b", query)
    if id_match:
        metadata["id"] = id_match.group()

    # Adjusted timestamp extraction to handle fractional seconds
    timestamp_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", query
    )
    if timestamp_match:
        metadata["timestamp"] = timestamp_match.group()

    # URL extraction
    url_match = re.search(r"https?://[^\s]+", query)
    if url_match:
        metadata["url"] = url_match.group()

    return metadata


if __name__ == "__main__":
    test_queries = [
        "Show news published on 2022-05-13T04:00:00",
        "Give me last update news from 2022-05-13T04:00:00",
        "Show me article 1.6272172",
        "Get guidelines from https://cbc.radio-canada.ca/example",
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        extracted = extract_metadata_from_query(query)
        print(f"Extracted metadata: {extracted}\n")
