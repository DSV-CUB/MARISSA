import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from marissa.toolbox.creators import creator_gui
from marissa.gui import gui_project, dialog_list_choice, dialog_add_edit_segmentation, dialog_plot_data_contour, dialog_export_data
from marissa.modules.database import marissadb

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.setWindowTitle("MARISSA - Project Data : " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])

        self.btn_update.clicked.connect(self.btn_update_clicked)
        self.btn_icon_new_segmentation.clicked.connect(self.btn_icon_new_segmentation_clicked)
        self.btn_export.clicked.connect(self.btn_export_clicked)
        self.btn_import_directory.clicked.connect(self.btn_import_directory_clicked)
        self.btn_import_project.clicked.connect(self.btn_import_project_clicked)
        self.btn_import_external.clicked.connect(self.btn_import_external_clicked)
        self.btn_icon_delete.clicked.connect(self.btn_icon_delete_clicked)

        self.tbl_data.doubleClicked.connect(self.tbl_data_clickedDouble)
        self.tbl_segmentation.doubleClicked.connect(self.tbl_segmentation_clickedDouble)

        self.tabs_data.currentChanged.connect(self.tab_clicked)

        self.where = ["LIMIT 100"] * self.tabs_data.count()
        self.selected = [[]] * self.tabs_data.count()

        self.tab_clicked(0)
        return

    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return

    def tab_clicked(self, idx=None):
        self.lst_columns.clear()
        self.txt_where.setPlainText("")

        if idx is None:
            tab = self.tabs_data.currentWidget().objectName().replace("tab_", "")
            index = [self.tabs_data.widget(i).objectName() for i in range(self.tabs_data.count())].index("tab_" + tab)
        else:
            index = idx

        self.tabs_data.setCurrentIndex(index)
        tab = self.tabs_data.currentWidget().objectName().replace("tab_", "")

        self.btn_icon_new_segmentation.setVisible((True if tab == "segmentation" else False))
        self.lbl_import.setText("import " + tab + " from")

        tbl_info = self.configuration.project.select("SELECT * FROM pragma_table_info('tbl_" + tab + "');")

        if self.selected[index] == []:
            self.selected[index] = [(1 if not tbl_info[i][1] == "data" and not tbl_info[i][1] == "points" and not tbl_info[i][1] == "mask" else 0) for i in range(len(tbl_info))]

        select_columns = []

        for i in range(len(tbl_info)):
            self.lst_columns.addItem(tbl_info[i][1])

            if self.selected[index][i] == 1:
                select_columns.append(tbl_info[i][1])
                self.lst_columns.item(i).setSelected(True)
            else:
                self.lst_columns.item(i).setSelected(False)

        self.txt_where.setPlainText(self.where[index].strip())

        str_where = self.where[index].strip()
        if str_where != "" and not str_where.upper().startswith("ORDER BY") and not str_where.upper().startswith("WHERE") and not str_where.upper().startswith("LIMIT"):
            str_where = "WHERE " + str_where

        try:
            selection = self.configuration.project.select("SELECT " + ", ".join(select_columns) + " FROM tbl_" + tab + " " + str_where)
        except:
            self.show_dialog("There is a misstake in the where clause, please check again.", "Critical")
            selection = []
        self.tab_fill(select_columns, selection, eval("self." + self.tabs_data.widget(index).objectName().replace("tab_", "tbl_")))
        return

    def tab_fill(self, columns, selection, tbl_widget):
        tbl_widget.clear()
        tbl_widget.setRowCount(0)

        tbl_widget.setRowCount(len(selection))
        tbl_widget.setColumnCount(len(columns))
        tbl_widget.setHorizontalHeaderLabels(columns)
        #tbl_widget.setColumnHidden(0, True)

        for i in range(len(selection)):
            for j in range(len(selection[i])):
                tbl_widget.setItem(i, j, QtWidgets.QTableWidgetItem(str(selection[i][j])))

        tbl_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        return

    def btn_update_clicked(self):
        tab = self.tabs_data.currentWidget().objectName().replace("tab_", "")
        index = [self.tabs_data.widget(i).objectName() for i in range(self.tabs_data.count())].index("tab_" + tab)
        self.where[index] = self.txt_where.toPlainText()

        self.selected[index] = [0] * len(self.selected[index])
        selected = [self.lst_columns.row(si) for si in self.lst_columns.selectedItems()]
        for s in selected:
            self.selected[index][s] = 1
        self.selected[index][0] = 1
        self.tab_clicked()
        return

    def btn_icon_new_segmentation_clicked(self):
        dialog = dialog_add_edit_segmentation.GUI(self, self.configuration, "rule")
        if dialog.exec():
            self.tab_clicked()
        return

    def btn_export_clicked(self):
        dialog = dialog_export_data.GUI(self, self.configuration)
        dialog.exec()
        return

    def btn_import_directory_clicked(self):
        if self.tabs_data.currentWidget().objectName().replace("tab_", "") == "data":
            path_from = self.get_directory(None)
            if path_from is None or path_from == "":
                pass
            else:
                description = self.show_dialog("A description of the imported data is recommended in order to filter more easy for inclusion or exclusion in the training process, otherwise keep the field blanket.", "Input")
                counter = self.configuration.project.import_data(path_from, (None if (description is None or description == "") else description))
                if counter > 0:
                    self.show_dialog(str(counter) + " data was imported.", "Information")
                    self.tab_clicked()
                else:
                    self.show_dialog("No new data could be imported.", "Information")
        elif self.tabs_data.currentWidget().objectName().replace("tab_", "") == "segmentation":
            dialog = dialog_add_edit_segmentation.GUI(self, self.configuration, "import")
            if dialog.exec():
                self.tab_clicked()
        return

    def btn_import_project_clicked(self):
        projects = self.configuration.get_projects()
        if len(projects) == 0:
            self.show_dialog("No other projects available. Please create or import a new one.", "Information")
        else:
            dialog = dialog_list_choice.GUI(self, self.configuration, description="Please choose a project from where to import data from:", list=projects)
            if dialog.exec() and not dialog.result is None:
                self.import_extern(os.path.join(self.configuration.path_projects, dialog.result[0] + ".marissadb"))
        return

    def btn_import_external_clicked(self):
        self.import_extern(self.get_file(None, "MARISSA Project (*.marissadb)"))
        return

    def import_extern(self, path=None):
        if not path is None and not path == "":
            tab = self.tabs_data.currentWidget().objectName().replace("tab_", "")
            if tab == "segmentation":
                counter = self.configuration.project.select("SELECT COUNT(*) FROM tbl_segmentation")[0][0]
                self.configuration.project.execute_extern(path, "INSERT OR IGNORE INTO main.tbl_segmentation SELECT * FROM externdb.tbl_segmentation WHERE externdb.tbl_segmentation.SOPinstanceUID IN (SELECT SOPinstanceUID FROM main.tbl_data)")
                counter = self.configuration.project.select("SELECT COUNT(*) FROM tbl_segmentation")[0][0] - counter
            else:
                counter = self.configuration.project.import_from_extern(path, "tbl_" + tab, False)

            if counter > 0:
                self.show_dialog(str(counter) + " new imported. Please note that existing entries with the same ID were not overwritten.", "Information")
                self.tab_clicked()
            else:
                self.show_dialog("Nothing new to import.", "Information")
        return

    def btn_icon_delete_clicked(self):
        answer = self.show_dialog("Are you sure to delte all shown data?\nThis will not affect standardization. ", "Question")

        if answer == QtWidgets.QMessageBox.Yes:
            try:
                tab = self.tabs_data.currentWidget().objectName().replace("tab_", "")
                index = [self.tabs_data.widget(i).objectName() for i in range(self.tabs_data.count())].index("tab_" + tab)

                tbl_info = self.configuration.project.select("SELECT * FROM pragma_table_info('tbl_" + tab + "') WHERE pk;")
                pks = [tbl_info[i][1] for i in range(len(tbl_info))]

                str_where = self.where[index].strip()
                if str_where != "" and not str_where.upper().startswith("ORDER BY") and not str_where.upper().startswith("WHERE") and not str_where.upper().startswith("LIMIT"):
                    str_where = "WHERE " + str_where

                select = self.configuration.project.select("SELECT " + ", ".join(pks) + " FROM tbl_" + tab + " " + str_where)
                select = np.array(select).flatten().tolist()

                exec("self.configuration.project.delete_" + tab + "(select)")
                self.tab_clicked()
            except:
                return
        return

    def tbl_data_clickedDouble(self):
        ID = self.tbl_data.item(self.tbl_data.currentRow(), 0).text()
        dialog = dialog_plot_data_contour.GUI(self, self.configuration, SOPinstanceUID=ID, segmentationID=None)
        dialog.exec()
        return

    def tbl_segmentation_clickedDouble(self):
        ID = self.tbl_segmentation.item(self.tbl_segmentation.currentRow(), 0).text()
        dialog = dialog_plot_data_contour.GUI(self, self.configuration, SOPinstanceUID=None, segmentationID=ID)
        dialog.exec()
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
