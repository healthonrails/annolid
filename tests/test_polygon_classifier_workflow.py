import json
from pathlib import Path

import pandas as pd
import torch

from annolid.behavior.polygon_classifier_workflow import (
    build_polygon_feature_dataset,
    generate_polygon_points_csv,
    predict_polygon_classifier_csv,
    train_polygon_classifier,
)


def _write_labelme(path: Path, *, label: str) -> None:
    payload = {
        "version": "5.0.0",
        "flags": {"attack": label == "attack", "groom": label == "groom"},
        "shapes": [
            {
                "label": "intruder",
                "shape_type": "polygon",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
            {
                "label": "resident",
                "shape_type": "polygon",
                "points": [[20, 20], [30, 20], [30, 30], [20, 30]],
            },
        ],
        "imagePath": path.with_suffix(".png").name,
        "imageHeight": 40,
        "imageWidth": 40,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_polygon_feature_dataset_creates_train_and_test_csvs(tmp_path):
    train_seq = tmp_path / "train" / "video_a"
    test_seq = tmp_path / "test" / "video_b"
    train_seq.mkdir(parents=True)
    test_seq.mkdir(parents=True)
    _write_labelme(train_seq / "frame_000001.json", label="attack")
    _write_labelme(test_seq / "frame_000001.json", label="groom")

    outcome = build_polygon_feature_dataset(
        train_folder=tmp_path / "train",
        test_folder=tmp_path / "test",
        output_folder=tmp_path / "out",
        num_points=4,
    )

    assert Path(outcome.train_csv).is_file()
    assert Path(outcome.test_csv).is_file()
    assert outcome.train_rows == 1
    assert outcome.test_rows == 1
    assert outcome.labels == ("attack", "groom")

    train_df = pd.read_csv(outcome.train_csv)
    assert train_df.loc[0, "label"] == "attack"
    assert train_df.loc[0, "video"] == "video_a"
    assert "intruder_features" in train_df.columns


def test_generate_polygon_points_csv_merges_predicted_shapes_with_manual_labels(
    tmp_path,
):
    annotation_dir = tmp_path / "video_a"
    annotation_dir.mkdir()
    _write_labelme(annotation_dir / "video_a_000000003.json", label="attack")
    labels_csv = tmp_path / "manual_labels.csv"
    labels_csv.write_text(
        "Unnamed: 0,background,still,abdomen-move\n"
        "0,1,0,0\n"
        "1,1,0,0\n"
        "2,0,1,0\n"
        "3,0,0,1\n",
        encoding="utf-8",
    )

    outcome = generate_polygon_points_csv(
        annotation_dir=annotation_dir,
        label_csv=labels_csv,
        output_csv=tmp_path / "polygon_points.csv",
        num_points=4,
    )

    assert outcome.rows == 1
    assert outcome.labels == ("abdomen-move",)
    assert "intruder_features" in outcome.polygon_columns

    df = pd.read_csv(outcome.output_csv)
    assert df.loc[0, "frame_number"] == 3
    assert df.loc[0, "label"] == "abdomen-move"
    assert "resident_features" in df.columns


def test_generate_polygon_points_csv_includes_predicted_shapes_from_ndjson(tmp_path):
    annotation_dir = tmp_path / "video_a"
    annotation_dir.mkdir()
    _write_labelme(annotation_dir / "video_a_000000000.json", label="attack")
    (annotation_dir / "video_a_annotations.ndjson").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "frame": 0,
                        "shapes": [
                            {
                                "label": "manual",
                                "shape_type": "polygon",
                                "points": [[0, 0], [2, 0], [2, 2], [0, 2]],
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "frame": 1,
                        "imageHeight": 20,
                        "imageWidth": 20,
                        "shapes": [
                            {
                                "label": "fly2",
                                "points": [[1, 1], [5, 1], [5, 4], [1, 4]],
                                "motion_index": 3.5,
                                "annotation_source": "cutie_vos",
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels_csv = tmp_path / "manual_labels.csv"
    labels_csv.write_text(
        "frame_number,background,walk\n0,1,0\n1,0,1\n",
        encoding="utf-8",
    )

    outcome = generate_polygon_points_csv(
        annotation_dir=annotation_dir,
        label_csv=labels_csv,
        output_csv=tmp_path / "polygon_points.csv",
        num_points=4,
    )

    assert outcome.rows == 2
    assert outcome.labels == ("background", "walk")
    assert "fly2_features" in outcome.polygon_columns
    assert "manual_features" not in outcome.polygon_columns

    df = pd.read_csv(outcome.output_csv)
    ndjson_row = df[df["frame_number"] == 1].iloc[0]
    assert ndjson_row["frame"] == "video_a_000000001.json"
    assert ndjson_row["label"] == "walk"
    assert "fly2_motion_index" in df.columns
    assert ndjson_row["fly2_motion_index"] == 3.5


def test_train_polygon_classifier_handles_single_video_csv(tmp_path):
    rows = []
    for frame in range(6):
        rows.append(
            {
                "video": "video_a",
                "frame": f"video_a_{frame:09d}.json",
                "frame_number": frame,
                "label": "background" if frame < 3 else "walk",
                "fly2_features": [float(frame), 0.0, 1.0, 0.0, 1.0, 1.0],
                "fly2_area": 1.0,
                "fly2_centroid": [float(frame), 0.5],
                "fly2_perimeter": 4.0,
                "fly2_motion_index": float("nan") if frame == 0 else float(frame),
            }
        )
    csv_path = tmp_path / "polygon_points.csv"
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)

    with torch.no_grad():
        outcome = train_polygon_classifier(
            train_csv=csv_path,
            test_csv=csv_path,
            output_dir=tmp_path / "runs",
            num_epochs=1,
            batch_size=2,
            hidden_dim=16,
            window_size=3,
            device="cpu",
        )

    assert Path(outcome.checkpoint_path).is_file()
    assert Path(outcome.metrics_path).is_file()


def test_train_polygon_classifier_supports_tcn_model(tmp_path):
    rows = []
    for frame in range(8):
        rows.append(
            {
                "video": "video_a",
                "frame": f"video_a_{frame:09d}.json",
                "frame_number": frame,
                "label": "background" if frame < 4 else "walk",
                "fly2_features": [float(frame), 0.0, 1.0, 0.0, 1.0, 1.0],
                "fly2_area": 1.0,
                "fly2_centroid": [float(frame), 0.5],
                "fly2_perimeter": 4.0,
            }
        )
    csv_path = tmp_path / "polygon_points.csv"
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)

    with torch.no_grad():
        outcome = train_polygon_classifier(
            train_csv=csv_path,
            test_csv=csv_path,
            output_dir=tmp_path / "runs",
            model_type="tcn",
            num_epochs=1,
            batch_size=2,
            hidden_dim=16,
            num_residual_blocks=1,
            window_size=3,
            device="cpu",
        )

    assert outcome.model_type == "tcn"
    assert Path(outcome.checkpoint_path).name == "polygon_tcn_classifier_best.pt"
    assert Path(outcome.metrics_path).is_file()
    assert "NaN" not in Path(outcome.metrics_path).read_text(encoding="utf-8")
    tcn_features = pd.read_csv(
        Path(outcome.run_dir) / "tcn_inputs" / "train" / "video_a_features.csv"
    )
    assert not tcn_features.isna().any().any()

    inference = predict_polygon_classifier_csv(
        feature_csv=csv_path,
        checkpoint_path=outcome.checkpoint_path,
        output_csv=tmp_path / "predictions.csv",
        device="cpu",
    )

    assert inference.model_type == "tcn"
    assert inference.rows == len(rows)
    predictions = pd.read_csv(inference.output_csv)
    assert "predicted_label" in predictions.columns
    assert set(["prob_background", "prob_walk"]).issubset(predictions.columns)
