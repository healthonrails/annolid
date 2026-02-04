from qtpy import QtWidgets, QtCore, QtGui


class AiRectangleWidget(QtWidgets.QWidget):
    """
    Widget for AI Rectangle Prediction.
    """

    def __init__(self):
        """
        Initialize the AI Rectangle Widget.
        """
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface.
        """
        # Create widgets
        self._aiRectanglePrompt = QtWidgets.QLineEdit()
        self._aiRectanglePrompt.setMaxLength(50)
        self._aiRectanglePrompt.setFont(QtGui.QFont("Arial", 10))
        self._aiRectanglePrompt.setFixedWidth(170)

        aiRectangleLabel = QtWidgets.QLabel(self.tr("Text Prompt"))
        aiRectangleLabel.setAlignment(QtCore.Qt.AlignCenter)
        aiRectangleLabel.setFont(QtGui.QFont(None, 10))
        aiRectangleLabel.setFixedWidth(170)

        # Optional CountGD toggle (off by default)
        self._useCountGDCheckbox = QtWidgets.QCheckBox(self.tr("Use CountGD"))
        self._useCountGDCheckbox.setToolTip(
            self.tr(
                "Enable CountGD-based object counting in addition to "
                "GroundingDINO. This can improve counting of repeated "
                "objects but is slower and requires CountGD to be installed."
            )
        )
        self._useCountGDCheckbox.setChecked(False)

        aiRectangleLayout = QtWidgets.QVBoxLayout()
        aiRectangleLayout.addWidget(self._aiRectanglePrompt)
        aiRectangleLayout.addWidget(aiRectangleLabel)
        aiRectangleLayout.addWidget(self._useCountGDCheckbox)

        self.aiRectangleWidget = QtWidgets.QWidget()
        self.aiRectangleWidget.setLayout(aiRectangleLayout)

        self.aiRectangleAction = QtWidgets.QWidgetAction(self)
        self.aiRectangleAction.setDefaultWidget(self.aiRectangleWidget)

        # Main layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.aiRectangleWidget)
        self.setLayout(layout)
