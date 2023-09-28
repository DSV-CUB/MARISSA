import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from marissa.toolbox.creators import creator_gui
from marissa.toolbox.tools import tool_R
from marissa.gui import gui_project, dialog_list_choice, dialog_add_edit_segmentation, dialog_plot_data_contour, dialog_export_data
from marissa.toolbox.tools import tool_pydicom, tool_general

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))
        self.setWindowTitle("MARISSA - Project FAMD Analysis : " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])

        columns = ["SOPInstanceUID", "StudyinstanceUID", "seriesnumber", "instancenumber", "seriesdescription", "identifier", "age", "gender", "size", "weight", "description", "acquisitiondatetime", "timestamp"]
        selection = self.configuration.project.select("SELECT " + ",".join(columns) + " FROM tbl_data ORDER BY identifier, seriesnumber, instancenumber")
        self.tbl_data.clear()
        self.tbl_data.setRowCount(0)

        self.tbl_data.setRowCount(len(selection))
        self.tbl_data.setColumnCount(len(columns))
        self.tbl_data.setHorizontalHeaderLabels(columns)
        for i in range(len(selection)):
            for j in range(len(selection[i])):
                self.tbl_data.setItem(i, j, QtWidgets.QTableWidgetItem(str(selection[i][j])))
        self.tbl_data.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        self.btn_data_select_all.clicked.connect(self.btn_data_select_all_clicked)
        self.btn_data_deselect_all.clicked.connect(self.btn_data_deselect_all_clicked)
        self.btn_data_select_sql.clicked.connect(self.btn_data_select_sql_clicked)
        self.btn_data_deselect_sql.clicked.connect(self.btn_data_deselect_sql_clicked)

        selection = self.configuration.project.select("SELECT description FROM tbl_setup ORDER BY description")
        standardized_setups = [selection[i][0] for i in range(len(selection))]
        self.opt_setup.addItems(standardized_setups)

        self.btn_famd.clicked.connect(self.btn_famd_clicked)

        return

    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return

    def btn_data_select_all_clicked(self):
        self.tbl_data.selectAll()
        return

    def btn_data_deselect_all_clicked(self):
        self.tbl_data.clearSelection()
        return

    def btn_data_select_sql_clicked(self):
        self.update_tbl_data_selection("SELECT")
        return

    def btn_data_deselect_sql_clicked(self):
        self.update_tbl_data_selection("DESELECT")
        return

    def update_tbl_data_selection(self, command):
        where = self.txt_data_where.toPlainText().strip()
        if not where == "" and not where.upper().startswith("WHERE") and not where.upper().startswith("LIMIT") and not where.upper().startswith("ORDER BY") and not where.upper().startswith("GROUP BY"):
            where = "WHERE " + where

        try:
            selection = self.configuration.project.select("SELECT SOPInstanceUID FROM tbl_data " + where)
            if len(selection) == 0:
                self.show_dialog("The SQL string has 0 results.", "Information")
            else:
                IDs = [selection[i][0] for i in range(len(selection))]
                selected = np.unique([self.tbl_data.selectedIndexes()[i].row() for i in range(len(self.tbl_data.selectedIndexes()))]).astype(str).tolist()

                for i in range(self.tbl_data.rowCount()):
                    if self.tbl_data.item(i, 0).text() in IDs:
                        if command.upper() == "SELECT" and str(i) not in selected:
                            self.tbl_data.selectRow(i)
                        elif command.upper() == "DESELECT" and str(i) in selected:
                            self.tbl_data.selectRow(i)
        except:
            self.show_dialog("There is a missake in the WHERE-Clause. Please check it.", "Critical")
        return

    def btn_famd_clicked(self):
        try:
            data = [] # numpy array of parameter values
            columnnames = [] # parameter names

            # prepare data
            selected = np.unique([self.tbl_data.selectedIndexes()[i].row() for i in range(len(self.tbl_data.selectedIndexes()))]).astype(str).tolist()
            SOPinstanceUIDs = []
            for i in range(self.tbl_data.rowCount()):
                if str(i) in selected:
                    SOPinstanceUIDs.append(self.tbl_data.item(i, 0).text())

            if len(SOPinstanceUIDs) == 0:
                self.show_dialog("No data selected.", "Critical")
            else:
                # prepare parameters
                setupID = self.configuration.project.select("SELECT setupID FROM tbl_setup WHERE description = '" + self.opt_setup.currentText() + "'")[0][0]

                parameters = self.configuration.project.select("SELECT p.parameterID, p.VR, p.VM, p.formula FROM tbl_parameter AS p INNER JOIN tbl_match_setup_parameter AS msp ON p.parameterID = msp.parameterID WHERE msp.setupID = " + str(setupID) + " ORDER BY msp.ordering ASC")
                parametersID = np.concatenate([[parameters[i][0]] * parameters[i][2] for i in range(len(parameters))])
                parametersVR = np.concatenate([[parameters[i][1]] * parameters[i][2] for i in range(len(parameters))])
                parametersVM = np.concatenate([[parameters[i][2]] * parameters[i][2] for i in range(len(parameters))])
                parametersVMindex = np.concatenate([np.arange(parameters[i][2]).tolist() for i in range(len(parameters))])
                parametersFormula = np.concatenate([[parameters[i][3]] * parameters[i][2] for i in range(len(parameters))])

                parameters = np.transpose(np.vstack((parametersID, parametersVR, parametersVM, parametersVMindex, parametersFormula)))

                for i in range(len(parameters)):
                    if parameters[i][2] == 1:
                        columnnames.append(self.configuration.project.select("SELECT description FROM tbl_parameter WHERE parameterID = " + str(parameters[i][0]))[0][0])
                    else:
                        columnnames.append(self.configuration.project.select("SELECT description FROM tbl_parameter WHERE parameterID = " + str(parameters[i][0]))[0][0] + "_" + str(parameters[i][3]))

                path_out = self.get_directory(None)

                if path_out is None or path_out == "":
                    pass
                else:
                    # read data parameters
                    for i in range(len(SOPinstanceUIDs)):
                        dcm = self.configuration.project.get_data(SOPinstanceUIDs[i])[0] # used in eval formula
                        case = []
                        skip = False

                        for i in range(len(parametersFormula)):
                            formula = parameters[i][4]
                            try:
                                x = eval(formula)
                                if int(parameters[i][2]) > 1:
                                    x = x[int(parameters[i][3])]
                                x = tool_pydicom.get_value(x, parameters[i][1])
                                case.append(x)
                            except:
                                skip = True
                                break

                        if not skip:
                            data.append(case)

                if len(data) == 0:
                    self.show_dialog("None of the data can be used as at least one parameter is missing or parameter formula has a bug.", "Critical")
                else:
                    data = np.array(data, dtype=object)
                    #try:
                    famd = tool_R.Setup_FAMD()
                    famd.run(data, path_out, columnnames=columnnames)
                    self.show_dialog("FAMD analysis successfully performed. " + str(len(data)) + " of " + str(len(SOPinstanceUIDs)) + " datasets could be used.", "Information")
                    #except:
                    #    self.show_dialog("There is a problem with the R package connection. Please ensure to have R, the R packages FactoMineR and factoextra as well as all necessary Python site-packages installed.", "Critical")
        except:
            self.show_dialog("FAMD analysis failed. Either the data is not suitable or the is a problem with R. Please check installation of R and the packages FactoMineR and factoextra.", "Warning")
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
