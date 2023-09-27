import sys
import os
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_gui
from marissa.gui import gui_project, dialog_add_edit_setup, dialog_add_edit_parameter, dialog_list_choice

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.setWindowTitle("MARISSA - Project Setup : " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])

        self.btn_icon_new.clicked.connect(self.btn_icon_new_clicked)
        self.btn_icon_edit.clicked.connect(self.btn_icon_edit_clicked)
        self.btn_icon_delete.clicked.connect(self.btn_icon_delete_clicked)
        self.btn_import_project.clicked.connect(self.btn_import_project_clicked)
        self.btn_import_external.clicked.connect(self.btn_import_external_clicked)

        self.tabs_setup.currentChanged.connect(self.tab_clicked)

        self.tbl_setup.doubleClicked.connect(self.btn_icon_edit_clicked)
        self.tbl_parameter.doubleClicked.connect(self.btn_icon_edit_clicked)

        self.tab_clicked(0)
        return

    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return

    def tab_clicked(self, idx=None):
        if idx is None:
            tab = self.tabs_setup.currentWidget().objectName().replace("tab_", "")
            index = [self.tabs_setup.widget(i).objectName() for i in range(self.tabs_setup.count())].index("tab_" + tab)
        else:
            index = idx

        self.lbl_import.setText("import " + self.tabs_setup.currentWidget().objectName().replace("tab_", "") + " from")

        if self.tabs_setup.widget(index).objectName() == "tab_parameter":
            columns = "tbl_parameter"
            selection = self.configuration.project.select("SELECT * FROM tbl_parameter ORDER BY description COLLATE NOCASE ASC")
        elif self.tabs_setup.widget(index).objectName() == "tab_setup":
            columns = "tbl_setup"
            list_fields = []
            tbl_data_info = self.configuration.project.select("SELECT * FROM pragma_table_info('" + columns + "');")
            for i in range(len(tbl_data_info)):
                list_fields.append(tbl_data_info[i][1])
            list_fields.append("parameters")
            columns = list_fields
            #selection = self.configuration.project.select("SELECT * FROM tbl_setup ORDER BY description COLLATE NOCASE ASC")
            selection = self.configuration.project.select("SELECT s.*, p.parameterlist FROM tbl_setup AS s INNER JOIN (SELECT setupID, GROUP_CONCAT(description, ', ') AS parameterlist FROM (SELECT p1.setupID AS setupID, p2.description AS description FROM tbl_match_setup_parameter AS p1 INNER JOIN tbl_parameter as p2 ON p1.parameterID = p2.parameterID ORDER BY p1.ordering) GROUP BY setupID) AS p ON s.setupID = p.setupID ORDER BY description COLLATE NOCASE ASC")
        else:
            return

        self.tabs_setup.setCurrentIndex(index)
        self.tab_fill(columns, selection, eval("self." + self.tabs_setup.widget(index).objectName().replace("tab_", "tbl_")))

        return

    def tab_fill(self, columns, selection, tbl_widget):
        if type(columns) == str:
            list_fields = []
            tbl_data_info = self.configuration.project.select("SELECT * FROM pragma_table_info('" + columns + "');")
            for i in range(len(tbl_data_info)):
                list_fields.append(tbl_data_info[i][1])
        else:
            list_fields = columns


        tbl_widget.clear()
        tbl_widget.setRowCount(0)

        tbl_widget.setRowCount(len(selection))
        tbl_widget.setColumnCount(len(list_fields))
        tbl_widget.setHorizontalHeaderLabels(list_fields)
        tbl_widget.setColumnHidden(0, True)

        for i in range(len(selection)):
            for j in range(len(selection[0])):
                tbl_widget.setItem(i, j, QtWidgets.QTableWidgetItem(str(selection[i][j])))

        tbl_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        return

    def btn_icon_new_clicked(self):
        tab = self.tabs_setup.currentWidget().objectName().replace("tab_", "")
        dialog = eval("dialog_add_edit_" + tab + ".GUI(self, self.configuration, None)")
        if dialog.exec():
            self.tab_clicked()
        return

    def btn_icon_edit_clicked(self):
        try:
            tab = self.tabs_setup.currentWidget().objectName().replace("tab_", "")
            row = eval("self." + self.tabs_setup.currentWidget().objectName().replace("tab_", "tbl_") + ".currentRow()")
            id = eval("self." + self.tabs_setup.currentWidget().objectName().replace("tab_", "tbl_") + ".item(" + str(row) + ", 0).text()")
            dialog = eval("dialog_add_edit_" + tab + ".GUI(self, self.configuration, id)")
            if dialog.exec():
                self.tab_clicked()
        except:
            pass
        return

    def btn_icon_delete_clicked(self):
        try:
            tab = self.tabs_setup.currentWidget().objectName().replace("tab_", "")
            row = eval("self." + self.tabs_setup.currentWidget().objectName().replace("tab_", "tbl_") + ".currentRow()")
            id = str(eval("self." + self.tabs_setup.currentWidget().objectName().replace("tab_", "tbl_") + ".item(" + str(row) + ", 0).text()"))
            description = str(eval("self." + self.tabs_setup.currentWidget().objectName().replace("tab_", "tbl_") + ".item(" + str(row) + ", 1).text()"))

            answer = self.show_dialog("Deleting '" + description + "' from tbl_" + tab + " will NOT delete standardizations that were performed with that " + tab + ". Are you sure to proceed?", "Question")

            if answer == QtWidgets.QMessageBox.Yes:
                exec("self.configuration.project.delete_" + tab + "(id)")
                self.tab_clicked()
        except:
            pass
        return

    def btn_import_external_clicked(self):
        self.import_extern(self.get_file(None, "MARISSA project (*.marissadb)"))
        return

    def btn_import_project_clicked(self):
        projects = self.configuration.get_projects()
        if len(projects) == 0:
            self.show_dialog("No other projects available.", "Information")
        else:
            dialog = dialog_list_choice.GUI(self, self.configuration, description="Please choose a project from where to import data for " + self.tabs_setup.currentWidget().objectName().replace("tab_", "tbl_") + ":", list=projects)
            if dialog.exec() and not dialog.result is None:
                self.import_extern(os.path.join(self.configuration.path_projects, dialog.result[0] + ".marissadb"))
        return

    def import_extern(self, path=None):
        if not path is None and not path == "":
            tab = self.tabs_setup.currentWidget().objectName().replace("tab_", "")
            if tab == "setup":
                counter = self.configuration.project.import_from_extern(path, "tbl_" + tab, False)
                self.configuration.project.execute_extern(path, "INSERT OR IGNORE INTO main.tbl_parameter SELECT * FROM externdb.tbl_parameter WHERE parameterID IN (SELECT parameterID FROM externdb.tbl_match_setup_parameter)")
                self.configuration.project.execute_extern(path, "INSERT OR IGNORE INTO main.tbl_match_setup_parameter SELECT * FROM  externdb.tbl_match_setup_parameter WHERE setupID IN (SELECT setupID FROM main.tbl_setup WHERE setupID NOT IN (SELECT setupID FROM main.tbl_match_setup_parameter))")
            else:
                counter = self.configuration.project.import_from_extern(path, "tbl_" + tab, False)

            if counter > 0:
                self.show_dialog(str(counter) + " new imported. Please note that existing entries with the same ID were not overwritten.", "Information")
                self.tab_clicked()
            else:
                self.show_dialog("Nothing new to import.", "Information")
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
