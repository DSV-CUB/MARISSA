import sys
import os
import numpy as np
from PyQt5 import QtWidgets

from marissa.gui import dialog_list_choice
from marissa.toolbox.creators import creator_dialog
from marissa.toolbox.tools import tool_general

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None, id=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.btn_icon_new.clicked.connect(self.btn_icon_new_clicked)
        self.btn_icon_edit.clicked.connect(self.btn_icon_edit_clicked)
        self.btn_copy.clicked.connect(self.btn_copy_clicked)
        self.btn_parameter_include.clicked.connect(self.btn_parameter_include_clicked)
        self.btn_parameter_exclude.clicked.connect(self.btn_parameter_exclude_clicked)
        self.btn_parameter_up.clicked.connect(self.btn_parameter_up_clicked)
        self.btn_parameter_down.clicked.connect(self.btn_parameter_down_clicked)

        self.id = id

        self.parameters = np.array(self.configuration.project.select("SELECT parameterID, description FROM tbl_parameter ORDER BY description COLLATE NOCASE ASC"))
        self.selected = np.zeros((len(self.parameters), )).astype(int)

        clusterings = [file.replace(".py", "") for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules", "clustering"))]
        clusterings.remove("__init__")
        clusterings.remove("__pycache__")
        self.opt_clustertype.addItems(clusterings)
        regressions = [file.replace(".py", "") for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules", "regression"))]
        regressions.remove("__init__")
        regressions.remove("__pycache__")
        self.opt_regressiontype.addItems(regressions)

        if self.id is None:
            self.btn_icon_new.setVisible(True)
            self.btn_icon_edit.setVisible(False)
        else:
            self.btn_icon_new.setVisible(False)
            self.btn_icon_edit.setVisible(True)

            description = self.configuration.project.select("SELECT description FROM tbl_setup WHERE setupID = " + self.id)[0][0]
            self.setWindowTitle("MARISSA - Edit Setup : " + description)
            self.txt_description.setText(description)
            self.sb_bins.setValue(self.configuration.project.select("SELECT bins FROM tbl_setup WHERE setupID = " + self.id)[0][0])
            self.opt_clustertype.setCurrentText(str(self.configuration.project.select("SELECT clustertype FROM tbl_setup WHERE setupID = " + self.id)[0][0]))
            self.opt_regressiontype.setCurrentText(str(self.configuration.project.select("SELECT regressiontype FROM tbl_setup WHERE setupID = " + self.id)[0][0]))
            self.opt_ytype.setCurrentText(str(self.configuration.project.select("SELECT ytype FROM tbl_setup WHERE setupID = " + self.id)[0][0]))
            self.opt_mode.setCurrentText(self.configuration.project.select("SELECT mode FROM tbl_setup WHERE setupID = " + self.id)[0][0])

            selected = np.array(self.configuration.project.select("SELECT parameterID FROM tbl_match_setup_parameter WHERE setupID = " + self.id + " ORDER BY ordering ASC")).flatten()
            for i in range(len(selected)):
                index = np.argwhere(self.parameters[:,0].astype(str) == str(selected[i])).flatten()[0]
                self.selected[index] = int(i+1)

        self.update_parameters()
        return

    def closeEvent(self, event):
        event.accept()
        return

    def update_parameters(self):
        self.lst_parameter.clear()
        self.lst_parameter_include.clear()

        selected = np.argwhere(self.selected>0).flatten()
        not_selected = np.argwhere(self.selected==0).flatten()

        # lst_parameter
        self.lst_parameter.addItems(self.parameters[not_selected, 1].tolist())

        # lst_parameter_include
        if len(selected)>0:
            for i in range(len(selected.flatten())):
                index = np.argwhere(self.selected == i+1).flatten()[0]
                self.lst_parameter_include.addItem(self.parameters[index, 1])

        return

    def btn_parameter_include_clicked(self):
        try:
            row = self.lst_parameter.currentRow()
            parameter = self.lst_parameter.currentItem().text()
            index = np.argwhere(self.parameters[:,1].flatten() == parameter).flatten()[0]
            self.selected[index] = int(np.max(self.selected) + 1)
            self.update_parameters()

            if row >= len(np.argwhere(self.selected==0)):
                row = row-1
            self.lst_parameter.setCurrentRow(row)

            self.lst_parameter_include.setCurrentRow(len(np.argwhere(self.selected>0))-1)
        except:
            pass
        return

    def btn_parameter_exclude_clicked(self):
        try:
            row = self.lst_parameter_include.currentRow()
            parameter = self.lst_parameter_include.currentItem().text()
            index = np.argwhere(self.parameters[:,1].flatten() == parameter).flatten()[0]
            position = self.selected[index]
            self.selected[index] = int(0)
            self.selected[self.selected>position] = self.selected[self.selected>position] - 1
            self.update_parameters()

            if row >= len(np.argwhere(self.selected)>0):
                row = row - 1
            self.lst_parameter_include.setCurrentRow(row)

            index = np.argwhere(self.parameters[:,1].flatten() == parameter).flatten()[0]
            row = index - np.count_nonzero(self.selected[:index+1])
            self.lst_parameter.setCurrentRow(row)

        except:
            pass

        return

    def btn_parameter_up_clicked(self):
        try:
            parameter = self.lst_parameter_include.currentItem().text()
            index = np.argwhere(self.parameters[:,1].flatten() == parameter).flatten()[0]
            position = self.selected[index]
            if position > 1:
                index2 = np.argwhere(self.selected == position - 1).flatten()[0]
                self.selected[index2] = position
                self.selected[index] = position - 1
                self.update_parameters()
                self.lst_parameter_include.setCurrentRow(position - 1 - 1) # -1 cause zero indexed
        except:
            pass
        return

    def btn_parameter_down_clicked(self):
        try:
            parameter = self.lst_parameter_include.currentItem().text()
            index = np.argwhere(self.parameters[:,1].flatten() == parameter).flatten()[0]
            position = self.selected[index]
            if position < np.max(self.selected):
                index2 = np.argwhere(self.selected == position + 1).flatten()[0]
                self.selected[index2] = position
                self.selected[index] = position + 1
                self.update_parameters()
                self.lst_parameter_include.setCurrentRow(position + 1 - 1) # -1 cause zero indexed
        except:
            pass
        return

    def get_values(self):
        description = tool_general.string_stripper(self.txt_description.text(), [])
        bins = str(self.sb_bins.value())
        clustertype = self.opt_clustertype.currentText()
        regressiontype = self.opt_regressiontype.currentText()
        ytype = self.opt_ytype.currentText()
        mode = self.opt_mode.currentText()

        parameterIDs = []
        for i in range(np.max(self.selected)):
            index = np.argwhere(self.selected == i+1).flatten()[0]
            parameterIDs.append(self.parameters[index, 0])

        if description == "":
            self.show_dialog("Please provide a valid description.", "Critical")
            result = False
        elif len(parameterIDs) == 0:
            self.show_dialog("Please select at least one parameter for this setup.", "Critical")
            result = False
        else:
            result = [description, bins, clustertype, regressiontype, ytype, mode, parameterIDs]
        return result


    def btn_icon_new_clicked(self):
        read = self.get_values()
        if read:
            worked = self.configuration.project.insert_setup(*read)
            if worked:
                self.accept()
            else:
                self.show_dialog("A setup with the same description already exist, please edit the existing one or save this with another description.", "Critical")
        return

    def btn_icon_edit_clicked(self):
        answer = self.show_dialog("Editing will NOT delete already performed standardization of MARISSA that used this setup. If the edited setup should be used, please re-train for it. Are you sure to proceed", "Question")
        read = self.get_values()
        if read and answer == QtWidgets.QMessageBox.Yes:
            worked = self.configuration.project.update_setup(self.id, *read)

            if worked:
                self.accept()
            else:
                self.show_dialog("Editing not possible, potentially the provided novel description already exists for another setup.", "Critical")
        return

    def btn_copy_clicked(self):
        choice = np.array(self.configuration.project.select("SELECT description FROM tbl_setup")).flatten().tolist()

        if len(choice) == 0:
            self.show_dialog("No other setup.", "Information")
        else:
            dialog = dialog_list_choice.GUI(self, self.configuration, description="Please choose a project from where to import data from:", list=choice)
            if dialog.exec():
                self.selected = np.zeros((len(self.parameters), )).astype(int)
                copyid = str(self.configuration.project.select("SELECT setupID FROM tbl_setup WHERE description = '" + dialog.result[0] + "'")[0][0])

                self.sb_bins.setValue(self.configuration.project.select("SELECT bins FROM tbl_setup WHERE setupID = " + copyid)[0][0])
                self.opt_clustertype.setCurrentText(str(self.configuration.project.select("SELECT clustertype FROM tbl_setup WHERE setupID = " + copyid)[0][0]))
                self.opt_regressiontype.setCurrentText(str(self.configuration.project.select("SELECT regressiontype FROM tbl_setup WHERE setupID = " + copyid)[0][0]))
                self.opt_ytype.setCurrentText(str(self.configuration.project.select("SELECT ytype FROM tbl_setup WHERE setupID = " + copyid)[0][0]))
                self.opt_mode.setCurrentText(self.configuration.project.select("SELECT mode FROM tbl_setup WHERE setupID = " + copyid)[0][0])

                selected = np.array(self.configuration.project.select("SELECT parameterID FROM tbl_match_setup_parameter WHERE setupID = " + copyid + " ORDER BY ordering ASC")).flatten()
                for i in range(len(selected)):
                    index = np.argwhere(self.parameters[:,0].astype(str) == str(selected[i])).flatten()[0]
                    self.selected[index] = int(i+1)
                self.update_parameters()
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
