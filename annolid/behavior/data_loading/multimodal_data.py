import json
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[Dict[str, Any]]: List of data entries as dictionaries.
    """
    try:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
        return []


def transform_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single entry to the desired format, adding a <video> token in the user's message.

    Args:
        entry (Dict[str, Any]): Original data entry.

    Returns:
        Dict[str, Any]: Transformed data entry with <video> token.
    """
    return {
        "messages": [
            {"content": f"<video> {entry['query']}", "role": "user"},
            {"content": entry["response"], "role": "assistant"}
        ],
        "videos": entry["videos"]
    }


def transform_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform a dataset to the desired format.

    Args:
        dataset (List[Dict[str, Any]]): Original dataset.

    Returns:
        List[Dict[str, Any]]: Transformed dataset.
    """
    return [transform_entry(entry) for entry in dataset]


def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save a list of dictionaries to a JSON file.

    Args:
        data (List[Dict[str, Any]]): Data to save.
        file_path (str): Path to the output JSON file.
    """
    try:
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile, indent=2)
        print(f"Transformed dataset saved to {file_path}")
    except IOError:
        print(f"Error: Unable to save file at {file_path}")


def main(input_path: str, output_path: str) -> None:
    """
    Main function to load, transform, and save the dataset.

    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output JSON file.
    """
    # Load dataset
    dataset = load_jsonl(input_path)
    if not dataset:
        return

    # Transform dataset
    transformed_data = transform_dataset(dataset)

    # Save transformed dataset
    save_json(transformed_data, output_path)


if __name__ == '__main__':
    # Paths to input and output files
    input_path = '/data/test_video_annotations.jsonl'
    output_path = '/data/transformed_video_annotations.json'

    # Run the main process
    main(input_path, output_path)
