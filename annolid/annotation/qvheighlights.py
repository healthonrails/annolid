import csv
import json
from annolid.annotation.timestamps import (
    convert_timestamp_to_seconds,
    convert_time_to_frame_number,
)


def write_jsonl_file(dataset, output_file):
    with open(output_file, "w") as file:
        for annotation in dataset:
            json.dump(annotation, file)
            file.write("\n")


def convert_csv_to_json(csv_file, query=None, output_file_path="output.jsonl"):
    """
    Convert a CSV file containing event start and event end annotations to a JSONL file.

    Args:
        csv_file (str): The path to the CSV file.
        query (str, optional): The query for the annotations. Defaults to None.
        output_file_path (str, optional): The path to the output JSONL file. Defaults to "output.jsonl".

    Returns:
        None

    Examples:
        >>> convert_csv_to_json('example.csv')
        # Converts the 'example.csv' file to 'output.jsonl' without a query field

        >>> convert_csv_to_json('example.csv', query='A family is playing basketball together on a green court outside.')
        # Converts the 'example.csv' file to 'output.jsonl' with the provided query field

        >>> convert_csv_to_json('example.csv', output_file_path='annotations.jsonl')
        # Converts the 'example.csv' file to 'annotations.jsonl' without a query field

        Below is an example of the annotation:

        {
            "qid": 8737,
            "query": "A family is playing basketball together on a green court outside.",
            "duration": 126,
            "vid": "bP5KfdFJzC4_660.0_810.0",
            "relevant_windows": [[0, 16]],
            "relevant_clip_ids": [0, 1, 2, 3, 4, 5, 6, 7],
            "saliency_scores": [[4, 1, 1], [4, 1, 1], [4, 2, 1], [4, 3, 2], [4, 3, 2], [4, 3, 3], [4, 3, 3], [4, 3, 2]]
        }
    """
    dataset = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Read and skip header line
        clip_id = 0
        for row in reader:
            timestamp, frame_number = row
            frame_number = frame_number.strip()
            if frame_number.endswith("'event_start')"):
                qid = int(frame_number.split(",")[0].strip("("))
                start_frame_id = convert_time_to_frame_number(timestamp)
                event_start = convert_timestamp_to_seconds(timestamp)
            elif frame_number.endswith("'event_end')"):
                event_end = convert_timestamp_to_seconds(timestamp)
                end_frame_id = convert_time_to_frame_number(timestamp)
                vid = f"{qid}_{event_start}_{event_end}"
                relevant_windows = [[start_frame_id, end_frame_id]]
                duration = end_frame_id - start_frame_id + 1
                relevant_clip_ids = [i for i in range(duration // 2)]
                saliency_scores = [[4] * 3 for _ in relevant_clip_ids]

                annotation = {
                    "qid": qid,
                    "query": query,
                    "duration": duration,
                    "vid": vid,
                    "relevant_clip_ids": relevant_clip_ids,
                    "relevant_windows": relevant_windows,
                    "saliency_scores": saliency_scores,
                }
                dataset.append(annotation)

                clip_id += 1

    # Write the dataset to a JSONL file
    write_jsonl_file(dataset, output_file_path)
    return dataset
