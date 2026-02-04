#!/usr/bin/env python3
import json
import numpy as np
import argparse
import logging


def resample_polygon(points, num_points):
    if not points:
        # Return an empty list or handle as needed if there are no points.
        return []
    points = np.array(points)
    if points.shape[0] == 0:
        return []

    # Ensure the polygon is closed (first and last point are the same)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Compute Euclidean distances between consecutive points
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumulative[-1]

    # Generate target distances equally spaced along the perimeter.
    # omit last to avoid duplicate
    target_distances = np.linspace(0, total_length, num_points + 1)[:-1]

    new_points = []
    for t in target_distances:
        i = np.searchsorted(cumulative, t) - 1
        if i >= len(points) - 1:
            i = len(points) - 2
        t0 = cumulative[i]
        t1 = cumulative[i + 1]
        ratio = (t - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
        new_point = points[i] + ratio * (points[i + 1] - points[i])
        new_points.append(new_point.tolist())
    return new_points


def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Reading input JSON file: %s", args.input)

    # Read the input JSON file
    with open(args.input, "r") as f:
        data = json.load(f)

    # List to hold the new, resampled polygon shapes
    resampled_shapes = []

    # Process each shape: if it's a polygon, resample its points
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "polygon":
            original_points = shape.get("points", [])
            if len(original_points) < 2:
                logging.warning(
                    "Skipping shape '%s' because it has fewer than 2 points.",
                    shape.get("label", "unknown"),
                )
                continue

            logging.info(
                "Resampling shape '%s' with %d original points.",
                shape.get("label", "unknown"),
                len(original_points),
            )
            new_points = resample_polygon(original_points, args.points)

            # Create a new shape with a modified label
            new_shape = shape.copy()
            new_shape["label"] = shape.get("label", "") + "_resampled"
            new_shape["points"] = new_points
            resampled_shapes.append(new_shape)

    # Append the resampled shapes to the original shapes list
    logging.info("Adding %d resampled shapes to JSON.", len(resampled_shapes))
    data["shapes"].extend(resampled_shapes)

    # Save the updated JSON to the output file
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    logging.info("Resampled polygons saved to %s", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample polygon points in a JSON file."
    )
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output", required=True, help="Path to the output JSON file.")
    parser.add_argument(
        "--points",
        type=int,
        required=True,
        help="Desired number of points per polygon.",
    )

    args = parser.parse_args()
    main(args)
