import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from matplotlib.ticker import ScalarFormatter


def analyze_motion_index(csv_path: str, output_dir: str = '.'):
    """
    Performs an analysis on a motion tracking CSV file,
    generating a clear dashboard and actionable outlier event summaries.

    This script produces:
    1. A detailed console report with statistics and outlier event bouts (start/end frames).
    2. A two-panel plot:
       - An Ethogram showing the activity profile over time, with top events annotated.
       - A Bar Chart summarizing the animal's "behavioral budget".

    Args:
        csv_path (str): The path to the tracked CSV file.
        output_dir (str): Directory to save the output analysis plot.
    """
    # --- 1. Load and Prepare Data ---
    print(f"🔬 Starting Analysis for: {os.path.basename(csv_path)}")
    print("-" * 60)

    try:
        df = pd.read_csv(csv_path)
        df['motion_index'] = pd.to_numeric(df['motion_index'], errors='coerce')
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {csv_path}")
        return

    df_filtered = df[df['motion_index'] >= 0].dropna(
        subset=['motion_index']).copy()

    if df_filtered.empty:
        print("⚠️ No valid motion data (motion_index >= 0) found to analyze.")
        return

    total_frames = len(df_filtered)
    print(
        f"✅ Data Loaded: Analyzing {total_frames} frames with valid motion data.")

    df_filtered['plot_motion_index'] = df_filtered['motion_index'].replace(
        0, 1e-9)

    # --- 2. Quantify Motion and Classify Behavioral States ---
    print("\n--- Motion Data Characterization ---")
    resting_threshold = 0.01
    high_activity_threshold = df_filtered['motion_index'].quantile(0.95)

    exact_zeros = df_filtered[df_filtered['motion_index'] == 0.0]
    resting_frames = df_filtered[df_filtered['motion_index']
                                 < resting_threshold]

    exact_zero_pct = len(exact_zeros) / total_frames * 100
    resting_pct = len(resting_frames) / total_frames * 100

    print(
        f"Frames with EXACTLY ZERO motion: {len(exact_zeros)} ({exact_zero_pct:.2f}%)")
    print(
        f"Frames classified as RESTING (motion < {resting_threshold}): {len(resting_frames)} ({resting_pct:.2f}%)")
    print(
        f"High Activity Threshold (95th percentile): {high_activity_threshold:.4f}")

    def classify_motion(motion_index):
        if motion_index < resting_threshold:
            return 'Resting'
        elif motion_index >= high_activity_threshold:
            return 'High Activity'
        else:
            return 'Moderate Activity'

    df_filtered['behavioral_state'] = df_filtered['motion_index'].apply(
        classify_motion)

    state_distribution = df_filtered['behavioral_state'].value_counts(
        normalize=True).mul(100).reindex(['Resting', 'Moderate Activity', 'High Activity'])

    print("\n--- Behavioral Budget (% of Time) ---")
    print(state_distribution.to_string())

    # **************************************************************************
    # *** Outlier Event Bout Detection and Reporting ***
    # **************************************************************************
    print("\n--- Outlier Motion Event Detection ---")
    outlier_threshold = high_activity_threshold * 1.5

    outlier_df = df_filtered[df_filtered['motion_index']
                             > outlier_threshold].copy()

    recommendations = pd.DataFrame()
    if not outlier_df.empty:
        # Identify contiguous blocks of outlier frames to define distinct events
        outlier_df['event_block'] = (
            outlier_df['frame_number'].diff() > 1).cumsum()

        # Aggregate each event block to find start, end, peak frame, and peak motion
        event_summary = outlier_df.groupby('event_block').agg(
            start_frame=('frame_number', 'min'),
            end_frame=('frame_number', 'max'),
            peak_motion=('motion_index', 'max')
        ).reset_index()

        # Find the frame number associated with the peak motion for each event
        peak_frames = outlier_df.loc[outlier_df.groupby(
            'event_block')['motion_index'].idxmax()]
        event_summary = event_summary.merge(
            peak_frames[['event_block', 'frame_number']], on='event_block')
        event_summary.rename(
            columns={'frame_number': 'peak_frame'}, inplace=True)

        # Sort events by intensity (most extreme first)
        recommendations = event_summary.sort_values(
            'peak_motion', ascending=False).head(10)
        recommendations = recommendations[[
            'start_frame', 'end_frame', 'peak_frame', 'peak_motion']]

        print(
            f"Detected {len(event_summary)} significant motion bouts (motion > {outlier_threshold:.2f}).")
        print("Top 10 most intense event bouts recommended for review:")
        print(recommendations.to_string(index=False))
    else:
        print("No significant outlier motion events detected above the threshold.")

    # --- 4. Generate High-Quality Visualizations ---
    print("\n📊 Generating the scientific dashboard...")
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1))

    fig.suptitle(
        f'Behavioral Activity Dashboard: {os.path.basename(csv_path)}', fontsize=24, y=0.97)

    # === Panel 1: The Ethogram ===
    ax1 = fig.add_subplot(gs[0, 0])
    state_palette = {'Resting': '#2ca02c',
                     'Moderate Activity': '#ff7f0e', 'High Activity': '#d62728'}

    sns.scatterplot(ax=ax1, data=df_filtered, x='frame_number', y='plot_motion_index',
                    hue='behavioral_state', palette=state_palette, s=12,
                    alpha=0.8, linewidth=0, hue_order=['Resting', 'Moderate Activity', 'High Activity'],
                    legend='full')

    ax1.set_yscale('log')
    ax1.set_ylabel('Motion Index (Log Scale)')
    ax1.set_title('Ethogram: Activity Profile Over Time', fontsize=18, pad=15)
    ax1.set_xlabel('')
    ax1.yaxis.set_major_formatter(ScalarFormatter())

    if not recommendations.empty:
        for idx, row in recommendations.head(5).iterrows():
            frame = row['peak_frame']
            motion = row['peak_motion']
            ax1.annotate(f"Event at Frame {frame}",
                         xy=(frame, motion),
                         xytext=(frame, motion * 1.5),
                         arrowprops=dict(facecolor='black',
                                         shrink=0.05, width=1, headwidth=8),
                         ha='center', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

    ax1.legend(title='Behavioral State', loc='upper left', frameon=True)

    # === Panel 2: The Behavioral Budget ===
    ax2 = fig.add_subplot(gs[1, 0])
    sns.barplot(ax=ax2, x=state_distribution.index, y=state_distribution.values,
                palette=state_palette, order=['Resting', 'Moderate Activity', 'High Activity'])

    ax2.set_ylabel('% of Total Time')
    ax2.set_xlabel('Behavioral State')
    ax2.set_title('Behavioral Budget', fontsize=18, pad=15)
    ax2.set_ylim(0, 100)

    for index, value in enumerate(state_distribution):
        ax2.text(index, value + 2, f"{value:.1f}%",
                 ha='center', va='bottom', fontsize=14, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_filename = os.path.join(output_dir, os.path.splitext(
        os.path.basename(csv_path))[0] + '_dashboard.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Dashboard saved to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a communication-focused motion analysis dashboard.")
    parser.add_argument("csv_path", type=str,
                        help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_dir", type=str,
                        default=".", help="Directory to save the analysis plot.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    analyze_motion_index(args.csv_path, args.output_dir)


if __name__ == '__main__':
    main()
