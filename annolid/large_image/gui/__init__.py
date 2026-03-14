from annolid.large_image.gui.status_overlay import LargeImageStatusOverlay

__all__ = ["LargeImageStatusOverlay", "TiledImageView", "ViewerLayerDockWidget"]


def __getattr__(name: str):
    if name == "TiledImageView":
        from annolid.gui.widgets.tiled_image_view import TiledImageView

        return TiledImageView
    if name == "ViewerLayerDockWidget":
        from annolid.gui.widgets.layer_dock import ViewerLayerDockWidget

        return ViewerLayerDockWidget
    raise AttributeError(name)
