import sys
from qtpy import QtCore
from qtpy import QtWidgets
from labelme.app import MainWindow
from labelme.utils import newIcon

__appname__ = 'Annolid'
__version__ = "1.0.0"


class AnnolidWindow(MainWindow):
    def __init__(self):
        super(AnnolidWindow,self).__init__()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))
    win = AnnolidWindow()

    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
