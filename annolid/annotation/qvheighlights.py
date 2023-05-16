import csv
import json
from annolid.annotation.timestamps import (convert_timestamp_to_seconds,
                                           convert_time_to_frame_number
                                           )


def write_jsonl_file(dataset, output_file):
    with open(output_file, 'w') as file:
        for annotation in dataset:
            json.dump(annotation, file)
            file.write('\n')


def convert_csv_to_json(csv_file,
                        query=None,
                        output_file_path="output.jsonl"):
    dataset = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header line
        clip_id = 0
        for row in reader:
            timestamp, frame_number = row
            frame_number = frame_number.strip()
            if frame_number.endswith("'event_start')"):
                qid = int(frame_number.split(',')[0].strip('('))
                start_frame_id = convert_time_to_frame_number(timestamp)
                event_start = convert_timestamp_to_seconds(timestamp)
            elif frame_number.endswith("'event_end')"):
                event_end = convert_timestamp_to_seconds(timestamp)
                end_frame_id = convert_time_to_frame_number(timestamp)
                vid = f"{qid}_{event_start}_{event_end}"
                relevant_windows = [[start_frame_id, end_frame_id]]
                duration = end_frame_id - start_frame_id + 1
                relevant_clip_ids = [i for i in range(duration//2)]
                saliency_scores = [[4] * 3 for _ in relevant_clip_ids]

                annotation = {
                    "qid": qid,
                    "query": query,
                    "duration": duration,
                    "vid": vid,
                    "relevant_clip_ids": relevant_clip_ids,
                    "relevant_windows": relevant_windows,
                    "saliency_scores": saliency_scores
                }
                dataset.append(annotation)

                clip_id += 1

    # Write the dataset to a JSONL file
    write_jsonl_file(dataset, output_file_path)
    return dataset
