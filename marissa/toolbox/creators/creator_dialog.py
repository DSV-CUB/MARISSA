import sys
import os

from PyQt5 import QtWidgets, QtGui, QtCore, uic
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
#QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False) #use highdpi icons

from marissa.gui import configuration
from marissa.toolbox.creators import creator_gui

class Inheritance(QtWidgets.QDialog):
    def __init__(self, parent=None, config=None, ui="dialog"):
        super().__init__(parent)

        if config is None:
            self.configuration = configuration.Setup()
        else:
            self.configuration = config

        self.ui = uic.loadUi(os.path.join(self.configuration.path, "gui", "designs", ui + ".ui"), self)

        try:
            self.setWindowIcon(QtGui.QIcon(self.configuration.path_icons["logo"]))
        except:
            pass

        self.ui.setWindowFlag(QtCore.Qt.CustomizeWindowHint, False)
        self.ui.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, False)
        self.ui.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, False)
        self.ui.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint, False)
        self.ui.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        #self.ui.setWindowState(self.ui.windowState() & ~QtCore.Qt.WindowContextHelpButtonHint & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        self.ui.activateWindow()

        list_mpl = []
        for parent_widget in self.ui.children():
            if "layout" in parent_widget.objectName().lower():
                for widget in parent_widget.children():
                    if widget.objectName().startswith("mpl_"):
                        oname = widget.objectName()
                        opos = parent_widget.layout().getItemPosition(parent_widget.layout().indexOf(widget))
                        parent_widget.layout().removeWidget(widget)
                        widget.close()
                        widget.deleteLater()
                        widget = creator_gui.QWidgetMatplotlib(parent_widget)#myDumpBox(self.ui.centralwidget)
                        widget.setObjectName(oname)
                        parent_widget.layout().addWidget(widget, opos[0], opos[1], opos[2], opos[3])
                        parent_widget.update()
                        exec("self." + oname + " = parent_widget.children()[-1]")
                        list_mpl.append(oname)

        try:
            for mpl in list_mpl:
                exec("self." + mpl + ".canvas.ax.imshow(matplotlib.pyplot.imread(self.configuration.path_icons[\"logo\"]))")
                exec("self." + mpl + ".canvas.ax.grid(False)")
                exec("self." + mpl + ".canvas.ax.xaxis.set_visible(False)")
                exec("self." + mpl + ".canvas.ax.yaxis.set_visible(False)")
                exec("self.mpl_image.canvas.ax.set_frame_on(False)")
                exec("self." + mpl + ".canvas.draw()")
        except:
            pass



        for parent_widget in self.ui.children():
            try:
                parent_widget.setFont(QtGui.QFont("Arial", 10))
            except:
                pass

            if parent_widget.objectName().startswith("tab"):
                for child_widget in parent_widget.children()[0].children():
                    if type(child_widget) == QtWidgets.QWidget:
                        for sub_child_widget in child_widget.children():
                            try:
                                sub_child_widget.setFont(QtGui.QFont("Arial", 10))
                            except:
                                pass

        icons = list(self.configuration.path_icons.keys())
        for parent_widget in self.ui.children():
            for icon in icons:
                if parent_widget.objectName().startswith("btn_icon_" + icon):
                    parent_widget.setIcon(QtGui.QIcon(self.configuration.path_icons[icon]))
                    parent_widget.setIconSize(parent_widget.size())
                    break
                elif parent_widget.objectName().startswith("lbl_icon_" + icon):
                    parent_widget.setPixmap(QtGui.QIcon(self.configuration.path_icons[icon]).pixmap(parent_widget.size()))
                    break

                elif parent_widget.objectName().startswith("tab"):
                    for child_widget in parent_widget.children()[0].children():
                        if type(child_widget) == QtWidgets.QWidget:
                            for sub_child_widget in child_widget.children():
                                for icon in icons:
                                    if sub_child_widget.objectName().startswith("btn_icon_" + icon):
                                        sub_child_widget.setIcon(QtGui.QIcon(self.configuration.path_icons[icon]))
                                        sub_child_widget.setIconSize(sub_child_widget.size())
                                        break
                                    elif sub_child_widget.objectName().startswith("lbl_icon_" + icon):
                                        sub_child_widget.setPixmap(QtGui.QIcon(self.configuration.path_icons[icon]).pixmap(sub_child_widget.size()))
                                        break
        return

    def get_directory(self, control):
        if isinstance(control, QtWidgets.QLabel):
            if control.text() == "":
                result = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a Directory', os.path.expanduser("~"))
            else:
                result = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a Directory', control.text())

            if not result == "":
                control.setText(result)
        else:
            result = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select a Directory', os.path.expanduser("~"))
        return result.replace("/", "\\")

    def get_file(self, control, filetype=None):
        if filetype is None:
            filetype = "All Files (*.*)"

        if isinstance(control, QtWidgets.QLabel):
            if control.text() == "":
                result = QtWidgets.QFileDialog.getOpenFileName(self, 'Select a File', os.path.expanduser("~"), filetype)
            else:
                result = QtWidgets.QFileDialog.getOpenFileName(self, 'Select a File', control.text(), filetype)

            if not result == "":
                control.setText(result[0])
        else:
            result = QtWidgets.QFileDialog.getOpenFileName(self, 'Select a File', os.path.expanduser("~"), filetype)
        return result[0].replace("/", "\\")

    def show_dialog(self, text, icon="Information", include_Cancel=False):
        if icon == "Input":
            answer, ok = QtWidgets.QInputDialog.getText(self, "Input Information", text + ":")
            if ok:
                result = answer
            else:
                result = False
        else:
            msg = QtWidgets.QMessageBox()
            try:
                msg.setIcon(eval("QtWidgets.QMessageBox." + icon.capitalize()))
            except:
                msg.setIcon(QtWidgets.QMessageBox.NoIcon)

            try:
                msg.setWindowIcon(QtGui.QIcon(self.configuration.path_icons["logo"]))
            except:
                pass

            msg.setText(text)
            #msg.setInformativeText("This is additional information")
            msg.setWindowTitle(icon.capitalize())
            #msg.setDetailedText("The details are as follows:")
            if icon.capitalize() == "Information" or icon.capitalize() == "Critical" or icon.capitalize() == "Warning":
                if include_Cancel:
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
                else:
                    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            elif icon.capitalize() == "Question":
                if include_Cancel:
                    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
                else:
                    msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

            result = msg.exec_()
        return result

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = Inheritance()
    gui_run.show()
    sys.exit(app.exec_())
