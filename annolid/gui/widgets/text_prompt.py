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
        self._aiRectanglePrompt.setFixedWidth(200)

        aiRectangleLabel = QtWidgets.QLabel(self.tr("AI Rectangle Prompt"))
        aiRectangleLabel.setAlignment(QtCore.Qt.AlignCenter)
        aiRectangleLabel.setFont(QtGui.QFont(None, 10))
        aiRectangleLabel.setFixedWidth(200)

        aiRectangleLayout = QtWidgets.QVBoxLayout()
        aiRectangleLayout.addWidget(self._aiRectanglePrompt)
        aiRectangleLayout.addWidget(aiRectangleLabel)

        self.aiRectangleWidget = QtWidgets.QWidget()
        self.aiRectangleWidget.setLayout(aiRectangleLayout)

        self.aiRectangleAction = QtWidgets.QWidgetAction(self)
        self.aiRectangleAction.setDefaultWidget(self.aiRectangleWidget)

        # Main layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.aiRectangleWidget)
        self.setLayout(layout)
