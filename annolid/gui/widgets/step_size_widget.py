from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class StepSizeWidget(QtWidgets.QSpinBox):
    def __init__(self, value=1):
        super(StepSizeWidget, self).__init__()
        self.setRange(-1000, 1000)
        self.setValue(value)
        self.setToolTip("Video Step Size")
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(StepSizeWidget, self).minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QtCore.QSize(width, height)
