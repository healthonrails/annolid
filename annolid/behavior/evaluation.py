import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


def _load_data(predicted_segments_path, manual_labels_path):
    """Loads and preprocesses prediction and manual label data."""
    try:
        predicted_df = pd.read_csv(predicted_segments_path)
        manual_df = pd.read_csv(manual_labels_path)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find one or both CSV files.")
    except pd.errors.ParserError:  # Handle potential parsing errors
        raise pd.errors.ParserError(
            "Error parsing CSV file(s). Check the format.")

    # Convert 'Recording time' to numeric and handle errors
    for df in [predicted_df, manual_df]:
        df['Recording time'] = pd.to_numeric(
            df['Recording time'], errors='coerce')
        # Remove rows with invalid times
        df.dropna(subset=['Recording time'], inplace=True)
    return predicted_df, manual_df


def _create_time_intervals(start_time, end_time, interval_duration):
    """Creates a DataFrame with fixed time intervals."""
    return pd.DataFrame({'Recording time': np.arange(start_time, end_time + interval_duration, interval_duration)})


def _align_behaviors_to_intervals(df, intervals_df, tolerance):
    """Aligns behavior labels to the nearest time interval."""
    merged_df = intervals_df.copy()
    merged_df['Behavior'] = 'none of the above'

    for _, row in df.iterrows():
        closest_time = intervals_df['Recording time'].iloc[(
            intervals_df['Recording time'] - row['Recording time']).abs().argsort()[0]]
        if abs(closest_time - row['Recording time']) <= tolerance:
            merged_df.loc[merged_df['Recording time'] ==
                          closest_time, 'Behavior'] = row['Behavior']
    return merged_df


def _calculate_metrics(y_true, y_pred):
    """Calculates accuracy, precision, recall, and F1-score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.unique(y_true))
    results_df = pd.DataFrame({
        'Behavior': np.unique(y_true),
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    return accuracy, results_df


def evaluate_behavior_predictions(predicted_segments_path, manual_labels_path, interval_duration=3):
    """Evaluates behavior predictions against manual labels using a fixed interval approach."""

    predicted_df, manual_df = _load_data(
        predicted_segments_path, manual_labels_path)

    start_time = min(predicted_df['Recording time'].min(
    ), manual_df['Recording time'].min())
    end_time = max(predicted_df['Recording time'].max(),
                   manual_df['Recording time'].max())

    intervals_df = _create_time_intervals(
        start_time, end_time, interval_duration)

    tolerance = interval_duration / 2
    merged_manual = _align_behaviors_to_intervals(
        manual_df, intervals_df, tolerance)
    merged_predicted = _align_behaviors_to_intervals(
        predicted_df, intervals_df, tolerance).rename(columns={"Behavior": "Behavior_predicted"})

    merged_df = pd.merge(merged_manual, merged_predicted,
                         on="Recording time", how="outer")

    y_true = merged_df['Behavior']
    y_pred = merged_df['Behavior_predicted']

    accuracy, results_df = _calculate_metrics(y_true, y_pred)

    return accuracy, results_df, merged_df


def plot_confusion_matrix(y_true, y_pred):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


# Main execution block
if __name__ == "__main__":
    predicted_path = '/content/predictions.csv'
    manual_path = '/content/ground_truth_annotations.csv'
    try:
        accuracy, results, merged_df = evaluate_behavior_predictions(
            predicted_path, manual_path)
        print(f"Overall Accuracy: {accuracy}")
        print("\nPer-Behavior Metrics:")
        print(results)
        plot_confusion_matrix(
            merged_df['Behavior'], merged_df['Behavior_predicted'])
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error: {e}")
