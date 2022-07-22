import pandas as pd
import matplotlib.pyplot as plt


def plot_trajactory(tracking_csv,
                    instance_name="mouse",
                    title="Trajectory",
                    xlabel="X position for instance centroid",
                    ylabel="Y position for instance centroid",
                    save_path='trajectory.png',
                    trajactory_color_style='b-'):
    """
    Plot a trajectory of a given instance.
    """
    df = pd.read_csv(tracking_csv)
    df = df[df['instance_name'] == instance_name]
    df.dropna(inplace=True)
    plt.figure(figsize=(10, 5))
    plt.title(f"{title} for {instance_name}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(df.cx, df.cy, trajactory_color_style)
    plt.savefig(save_path)
    plt.close()
