from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo


class Sam3VideoTrackingMultiplex(Sam3VideoTrackingMultiplexDemo):
    """
    Runtime entrypoint for SAM3 multiplex tracking in application inference.

    Keeps behavior compatible with existing multiplex tracking implementation while
    avoiding direct dependency on the `*Demo` class name in model wiring.
    """
