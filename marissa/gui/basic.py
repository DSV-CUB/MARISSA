import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore

from marissa.toolbox.creators import creator_gui
from marissa.gui import gui_project

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        return

    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return



if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
