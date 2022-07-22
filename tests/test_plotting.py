import os
import numpy as np
import pandas as pd
from annolid.postprocessing.plotting import plot_trajactory


def test_plot_trajactory():
    tracking_csv = '/tmp/tracking.csv'
    cx = np.random.randint(0, 100, size=100)
    cy = np.random.randint(0, 100, size=100)
    instance_name = ['mouse'] * 100
    df = pd.DataFrame({'cx': cx,
                       'cy': cy,
                       'instance_name': instance_name})

    df.to_csv(tracking_csv, index=False)

    plot_trajactory(tracking_csv, instance_name="mouse",
                    title="Trajectory",
                    xlabel="X position for instance centroid",
                    ylabel="Y position for instance centroid",
                    save_path='/tmp/trajectory.png',
                    trajactory_color_style='b-')
    assert os.path.isfile('/tmp/trajectory.png')
