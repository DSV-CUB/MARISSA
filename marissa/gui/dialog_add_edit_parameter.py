import sys
import os
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_dialog
from marissa.toolbox.tools import tool_general, tool_pydicom

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None, id=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.btn_icon_new.clicked.connect(self.btn_icon_new_clicked)
        self.btn_icon_edit.clicked.connect(self.btn_icon_edit_clicked)

        self.id = id

        self.opt_VR.addItems(tool_pydicom.get_standard_VR())

        if self.id is None:
            self.btn_icon_new.setVisible(True)
            self.btn_icon_edit.setVisible(False)
        else:
            self.btn_icon_new.setVisible(False)
            self.btn_icon_edit.setVisible(True)

            description = self.configuration.project.select("SELECT description FROM tbl_parameter WHERE parameterID = " + self.id)[0][0]
            self.setWindowTitle("MARISSA - Edit Parameter : " + description)
            self.txt_description.setText(description)
            self.opt_VR.setCurrentText(str(self.configuration.project.select("SELECT VR FROM tbl_parameter WHERE parameterID = " + self.id)[0][0]))
            self.sb_VM.setValue(self.configuration.project.select("SELECT VM FROM tbl_parameter WHERE parameterID = " + self.id)[0][0])
            self.txt_formula.setPlainText(self.configuration.project.select("SELECT formula FROM tbl_parameter WHERE parameterID = " + self.id)[0][0])
        return

    def closeEvent(self, event):
        event.accept()
        return

    def get_values(self):
        description = tool_general.string_stripper(self.txt_description.text())
        VR = self.opt_VR.currentText()
        VM = str(self.sb_VM.value())
        formula = self.txt_formula.toPlainText().strip()

        if description == "":
            self.show_dialog("Please provide a valid description.", "Critical")
            result = False
        elif formula == "":
            self.show_dialog("Please provide a formula to calculate the parameter", "Critical")
            result = False
        else:
            result = [description, VR, VM, formula]
        return result

    def btn_icon_new_clicked(self):
        read = self.get_values()
        if read:
            worked = self.configuration.project.insert_parameter(*read)
            if worked:
                self.accept()
            else:
                self.show_dialog("A parameter with the same description already exist, please edit the existing one or save this with another description.", "Critical")
        return

    def btn_icon_edit_clicked(self):
        answer = self.show_dialog("Editing will NOT delete already performed standardization of MARISSA that used this parameter. If the edited parameter should be used, please re-train the setups. Are you sure to proceed?", "Question")
        read = self.get_values()
        if read and answer == QtWidgets.QMessageBox.Yes:
            worked = self.configuration.project.update_parameter(self.id, *read)

            if worked:
                self.accept()
            else:
                self.show_dialog("Editing not possible, potentially the provided novel description already exists for another setup.", "Critical")
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
