import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_trajactory(
    tracking_csv,
    instance_name="mouse",
    title="Trajectory",
    xlabel="X position for instance centroid",
    ylabel="Y position for instance centroid",
    save_path="trajectory.png",
    trajactory_color_style="b-",
):
    """
    Plot a trajectory of a given instance.
    """
    df = pd.read_csv(tracking_csv)
    df = df[df["instance_name"] == instance_name]
    df.dropna(inplace=True)
    # Use Agg canvas directly to avoid GUI backend teardown issues in tests.
    figure = Figure(figsize=(10, 5))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_title(f"{title} for {instance_name}")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.plot(df.cx, df.cy, trajactory_color_style)
    figure.savefig(save_path)
