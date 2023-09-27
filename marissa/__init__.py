import os
from PyQt5 import QtWidgets, QtCore
import sys
from marissa import gui, modules, toolbox



if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = gui.gui_start.GUI()
    gui_run.show()
    sys.exit(app.exec_())