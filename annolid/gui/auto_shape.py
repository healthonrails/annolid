class AutoShapes:
    """Manages automatic shape processing or generation."""

    def __init__(self, shapes, replace=True):
        """Initialize AutoShapes

        Args:
            shapes (List[Shape]): List of shapes to process or generate.
            replace (bool, optional): Replace all current shapes with new shapes.
                Defaults to True.
        """
        self.shapes = shapes
        self.replace = replace


class AutoShapesMode:
    """Represents different modes for handling or creating auto shapes."""

    OBJECT = "AUTOSHAPE_OBJECT"
    ADD = "AUTOSHAPE_ADD"
    REMOVE = "AUTOSHAPE_REMOVE"
    POINT = "point"
    RECTANGLE = "rectangle"

    def __init__(self, edit_mode, shape_type):
        """Initialize AutoShapesMode

        Args:
            edit_mode (str): Edit mode, can be AUTOSHAPE_ADD or AUTOSHAPE_REMOVE.
            shape_type (str): Type of shape, can be point or rectangle.
        """
        self.edit_mode = edit_mode
        self.shape_type = shape_type

    @staticmethod
    def get_default_mode():
        """Get the default mode."""
        return AutoShapesMode(AutoShapesMode.ADD, AutoShapesMode.POINT)

    # Compare 2 instances of AutoShapesMode
    def __eq__(self, other):
        if not isinstance(other, AutoShapesMode):
            return False
        return (
            self.edit_mode == other.edit_mode
            and self.shape_type == other.shape_type
        )
