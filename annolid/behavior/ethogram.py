import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_ethogram(
    data,
    subject_column="Subject",
    behavior_column="Behavior",
    start_column="Recording time",
    end_column="Recording time",
    duration_column=None,
    subject_colors=None,
    event_column="Event",
    behavior_colors=None,
):
    """
    Creates an ethogram visualization with each behavior per row for each subject.

    Args:
        data (pd.DataFrame): DataFrame with columns for subject, behavior, start time, and end time.
        subject_column (str): Name of the column containing subject identifiers.
        behavior_column (str): Name of the column containing behavior labels.
        start_column (str): Name of the column containing start times.
        end_column (str): Name of the column containing end times. If `duration_column` is specified, this will be ignored.
        duration_column (str): Name of the column indicating the duration of the behavior in seconds. If None, uses `end_column`.
        subject_colors (dict, optional): Dictionary mapping subject names to colors. Defaults to None (automatic colors).
        event_column (str, optional): Name of the column containing event labels (e.g., "state start", "state stop").
        behavior_colors (dict, optional): Dictionary mapping behavior labels to colors. Defaults to None (automatic colors).

    Returns:
        matplotlib.figure.Figure: The ethogram figure.

    Raises:
        ValueError: If the input data does not have the required columns.
    """

    required_cols = [subject_column, behavior_column, start_column]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {', '.join(required_cols)}")

    if duration_column is not None and duration_column not in data.columns:
        raise ValueError(f"Data must contain duration column: {duration_column}")

    if subject_colors is None:
        # Use a colormap for automatic colors
        subject_colors = plt.cm.get_cmap("Set1")

    fig, ax = plt.subplots(figsize=(15, 10))

    # Iterate through subjects
    for subject_index, subject in enumerate(data[subject_column].unique()):
        subject_data = data[data[subject_column] == subject]

        # Get unique behaviors for this subject
        behaviors = subject_data[behavior_column].unique()

        # Iterate through behaviors for the current subject
        for behavior_index, behavior in enumerate(behaviors):
            behavior_data = subject_data[subject_data[behavior_column] == behavior]

            # Calculate y_pos based on subject and behavior
            y_pos = subject_index * len(behaviors) + behavior_index

            for _, row in behavior_data.iterrows():
                start = row[start_column]
                if duration_column is None:
                    end = row[end_column]
                else:
                    end = start + row[duration_column]

                # Color handling (modified)
                if behavior_colors is not None:  # Use behavior colors if provided
                    color = behavior_colors.get(behavior, "gray")
                elif isinstance(subject_colors, dict):
                    color = subject_colors.get(subject, "gray")
                else:
                    # Use subject index for colormap
                    color = subject_colors(subject_index)

                rect = patches.Rectangle(
                    (start, y_pos - 0.25),
                    end - start,
                    0.5,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=color,
                    label=behavior,
                )
                ax.add_patch(rect)

                # Add event markers (triangles)
                if event_column in data.columns:
                    event = row[event_column]
                    if event == "state start":
                        ax.plot(
                            start, y_pos, marker="^", color="darkred", markersize=5
                        )  # Start marker
                    elif event == "state stop":
                        ax.plot(
                            end, y_pos, marker="v", color="darkgreen", markersize=5
                        )  # Stop marker

    # Set y-axis labels and limits (updated for behavior rows)
    y_ticks = []
    y_labels = []
    for subject_index, subject in enumerate(data[subject_column].unique()):
        behaviors = data[data[subject_column] == subject][behavior_column].unique()
        for behavior_index, behavior in enumerate(behaviors):
            y_ticks.append(subject_index * len(behaviors) + behavior_index)
            y_labels.append(f"{subject} - {behavior}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.5, len(y_ticks) - 0.5)

    # Set axis labels, title, limits and grid
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Subjects and Behaviors")  # updated y axis label
    ax.set_title("Ethogram")
    ax.set_xlim(
        0, data[start_column if duration_column is not None else end_column].max() + 10
    )  # Dynamic x-axis limit
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9)
    return fig


if __name__ == "__main__":
    df = pd.read_csv("example_behavior_anno.csv")
    df.head()
    # Using default colors from colormap:

    behavior_colors = {
        "Stimulus Mouse Sniffing FP Mouse": "lightcoral",
        "Nose-to-Nose Sniffing": "skyblue",
        "Nose-to-Flank Sniffing": "lightgreen",
        # ... other behavior colors ...
    }

    fig = create_ethogram(
        df,
        subject_column="Subject",
        behavior_column="Behavior",
        start_column="Recording time",
        duration_column="Trial time",
        event_column="Event",
        behavior_colors=behavior_colors,
    )

    plt.show()

    # Saving the figure:
    fig.savefig("ethogram.png")
