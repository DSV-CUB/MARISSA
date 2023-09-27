import sys
import os
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from marissa.toolbox.creators import creator_gui, creator_sqlitedb
from marissa.toolbox.tools import tool_general
from marissa.gui import gui_project, dialog_list_choice

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.setWindowTitle("MARISSA - Project Train : " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])
        self.btn_next.clicked.connect(self.btn_next_clicked)
        self.btn_previous.clicked.connect(self.btn_previous_clicked)

        # tab setup
        self.btn_icon_new_setup_previous.clicked.connect(self.btn_icon_new_setup_previous_clicked)
        self.btn_import_project.clicked.connect(self.btn_import_project_clicked)
        self.btn_import_external.clicked.connect(self.btn_import_external_clicked)

        # tab data
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

        # tab segmentation
        columns = ["segmentation"]
        selection = self.configuration.project.select("SELECT DISTINCT description, creator FROM tbl_segmentation ORDER BY description")
        selection = [selection[i][0] + " | " + selection[i][1] for i in range(len(selection))]
        self.tbl_segmentation.clear()
        self.tbl_segmentation.setRowCount(0)

        self.tbl_segmentation.setRowCount(len(selection))
        self.tbl_segmentation.setColumnCount(len(columns))
        self.tbl_segmentation.setHorizontalHeaderLabels(columns)
        for i in range(len(selection)):
            self.tbl_segmentation.setItem(i, 0, QtWidgets.QTableWidgetItem(str(selection[i])))
        self.tbl_segmentation.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        # tab train
        self.btn_train_default_reference.clicked.connect(self.btn_train_default_reference_clicked)
        self.btn_icon_new_train.clicked.connect(self.btn_icon_new_train_clicked)
        self.cb_manual_reference.toggled.connect(self.cb_reference_clicked)
        self.cb_automatic_reference.toggled.connect(self.cb_reference_clicked)

        # initialize
        self.update_setup()
        self.tab_update(0)

        self.reference_parameter_array = []
        self.reference_automatic = []
        self.reference_selected = []
        self.SOPinstanceUIDs = []
        return

    def update_setup(self):
        self.opt_setup.clear()
        selection = np.array(self.configuration.project.select("SELECT description FROM tbl_setup")).flatten()
        if len(selection) > 0:
            self.opt_setup.addItems(selection.tolist())
        return

    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return

    def btn_next_clicked(self):
        self.tab_update(int(self.tabs_train.currentIndex()) + 1)
        return

    def btn_previous_clicked(self):
        self.tab_update(int(self.tabs_train.currentIndex()) - 1)
        return

    def tab_update(self, goto_index):
        current_index = int(self.tabs_train.currentIndex())
        current_tab = self.tabs_train.widget(current_index).objectName()
        goto_tab = self.tabs_train.widget(goto_index).objectName()
        switch_tab = True

        setupID = str(self.configuration.project.select("SELECT setupID FROM tbl_setup WHERE description = '" + self.opt_setup.currentText() + "'")[0][0])

        # check if actions are valid and save changes
        if goto_index > current_index:
            if current_tab == "tab_setup":
                if self.opt_setup.currentText() == "":
                    self.show_dialog("No setup selected, please create a setup", "Critical")
                    switch_tab = False
                else:
                    # select data
                    selection = self.configuration.project.select("SELECT SOPinstanceUID FROM tbl_match_setup_data_segmentation WHERE setupID = " + setupID)
                    SOPinstanceUIDs = [selection[i][0] for i in range(len(selection))]

                    self.tbl_data.clearSelection()
                    if len(SOPinstanceUIDs) > 0:
                        for i in range(self.tbl_data.rowCount()):
                            if self.tbl_data.item(i, 0).text() in SOPinstanceUIDs:
                                self.tbl_data.selectRow(i)

                    # select segmentation
                    selection = self.configuration.project.select("SELECT description, creator FROM tbl_segmentation WHERE segmentationID IN (SELECT segmentationID FROM tbl_match_setup_data_segmentation WHERE setupID = " + setupID + ")")
                    segmentations = [selection[i][0] + " | " + selection[i][1] for i in range(len(selection))]

                    self.tbl_segmentation.clearSelection()
                    if len(segmentations) > 0:
                        for i in range(self.tbl_segmentation.rowCount()):
                            if self.tbl_segmentation.item(i, 0).text() in segmentations:
                                self.tbl_segmentation.selectRow(i)

            elif current_tab == "tab_data":
                row_indeces = []
                enough = False

                for i in range(len(self.tbl_data.selectedIndexes())):
                    row_indeces.append(self.tbl_data.selectedIndexes()[i].row())
                    if len(np.unique(row_indeces)) >= 2:
                        enough = True
                        break

                if not enough:
                    self.show_dialog("Not enough data selected, please select at least two dataset", "Critical")
                    switch_tab = False

            elif current_tab == "tab_segmentation":
                if len(self.tbl_segmentation.selectedIndexes()) == 0:
                    self.show_dialog("No segmentation selected, please select at least one", "Critical")
                    switch_tab = False
                else:
                    self.configuration.project.execute("DELETE FROM tbl_match_setup_data_segmentation WHERE setupID = " + setupID)

                    #selected = np.unique([self.tbl_data.selectedIndexes()[i].row() for i in range(len(self.tbl_data.selectedIndexes()))]).astype(str).tolist()
                    #SOPinstanceUIDs = []
                    #for i in range(self.tbl_data.rowCount()):
                    #    if str(i) in selected:
                    #        SOPinstanceUIDs.append(self.tbl_data.item(i, 0).text())

                    selected = np.unique([self.tbl_segmentation.selectedIndexes()[i].row() for i in range(len(self.tbl_segmentation.selectedIndexes()))]).astype(str).tolist()
                    for i in range(self.tbl_segmentation.rowCount()):
                        if str(i) in selected:
                            description, creator = self.tbl_segmentation.item(i, 0).text().split(" | ")
                            self.configuration.project.execute("INSERT INTO tbl_match_setup_data_segmentation SELECT " + setupID + ", SOPinstanceUID, segmentationID FROM tbl_segmentation WHERE description = '" + description + "' AND CREATOR = '" + creator + "' AND SOPinstanceUID IN ('" + "', '".join(self.SOPinstanceUIDs) + "')")

            elif current_tab == "tab_train":
                pass

        # switch to other tab
        if switch_tab:
            if goto_index > current_index:
                if goto_tab == "tab_setup":
                    pass
                elif goto_tab == "tab_data":
                    pass
                elif goto_tab == "tab_segmentation":
                    selected = np.unique([self.tbl_data.selectedIndexes()[i].row() for i in range(len(self.tbl_data.selectedIndexes()))]).astype(str).tolist()
                    self.SOPinstanceUIDs = []
                    for i in range(self.tbl_data.rowCount()):
                        if str(i) in selected:
                            self.SOPinstanceUIDs.append(self.tbl_data.item(i, 0).text())
                    selection = self.configuration.project.select("SELECT DISTINCT description, creator FROM tbl_segmentation WHERE SOPinstanceUID IN ('" + "', '".join(self.SOPinstanceUIDs)  + "')")

                    segmentations = [selection[i][0] + " | " + selection[i][1] for i in range(len(selection))]

                    for i in range(self.tbl_segmentation.rowCount()):
                        if not self.tbl_segmentation.item(i, 0).text() in segmentations:
                            self.tbl_segmentation.hideRow(i)
                        else:
                            self.tbl_segmentation.showRow(i)
                elif goto_tab == "tab_train":
                    #selected_data = len(np.unique([self.tbl_data.selectedIndexes()[i].row() for i in range(len(self.tbl_data.selectedIndexes()))]).astype(str).tolist())
                    selected_data = self.configuration.project.select("SELECT COUNT (DISTINCT SOPinstanceUID) FROM tbl_match_setup_data_segmentation WHERE setupID = " + str(setupID))[0][0]
                    #selected_segmentation = len(np.unique([self.tbl_segmentation.selectedIndexes()[i].row() for i in range(len(self.tbl_segmentation.selectedIndexes()))]).astype(str).tolist())
                    selected_segmentation = self.configuration.project.select("SELECT COUNT(*) FROM (SELECT DISTINCT description, creator FROM tbl_segmentation WHERE segmentationID IN (SELECT DISTINCT segmentationID FROM tbl_match_setup_data_segmentation WHERE setupID = " + str(setupID) + "))")[0][0]
                    selected_combine = self.configuration.project.select("SELECT COUNT(*) FROM tbl_match_setup_data_segmentation WHERE setupID = " + setupID)[0][0]

                    infotext = "For the training of the setup " + self.opt_setup.currentText() + " with"
                    infotext = infotext + "\nbins:\t" + str(self.configuration.project.select("SELECT bins FROM tbl_setup WHERE setupID = " + setupID)[0][0])
                    infotext = infotext + "\nclustertype:\t" + self.configuration.project.select("SELECT clustertype FROM tbl_setup WHERE setupID = " + setupID)[0][0]
                    infotext = infotext + "\nregressiontype:\t" + self.configuration.project.select("SELECT regressiontype FROM tbl_setup WHERE setupID = " + setupID)[0][0]
                    infotext = infotext + "\nytype:\t" + self.configuration.project.select("SELECT ytype FROM tbl_setup WHERE setupID = " + setupID)[0][0]
                    infotext = infotext + "\nmode:\t" + self.configuration.project.select("SELECT mode FROM tbl_setup WHERE setupID = " + setupID)[0][0]
                    infotext = infotext + "\n\n" + str(selected_data) + " individual DICOM data and " + str(selected_segmentation) + " different segmentations were chosen. "
                    infotext = infotext + "In total " + str(selected_combine) + " segmented DICOM data could be identified for training."
                    infotext = infotext + "\n\nIf you are sure, you can start the training with [run training]. Please keep in mind, that previous stored segmentations get overwritten. "
                    infotext = infotext + "This may take some time, please leave the GUI open until the training is finished otherwise all performed training gets lost."
                    self.lbl_train.setText(infotext)
                    self.tbl_reference_changed(-1)

                    self.tbl_data.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

            self.tabs_train.setTabEnabled(goto_index, True)
            self.tabs_train.setCurrentIndex(goto_index)
            for i in range(self.tabs_train.count()):
                if i != goto_index:
                    self.tabs_train.setTabEnabled(i, False)

            if goto_index == 0:
                self.btn_previous.setVisible(False)
            else:
                self.btn_previous.setVisible(True)

            if goto_index == self.tabs_train.count() - 1:
                self.btn_next.setVisible(False)
            else:
                self.btn_next.setVisible(True)
        return


    # tab setup
    def btn_icon_new_setup_previous_clicked(self):
        if self.opt_setup.currentText() == "":
            self.show_dialog("No setup selected, please create a setup.", "Critical")
        else:
            selection = self.configuration.project.select("SELECT setupID FROM tbl_standardization_setup WHERE description = '" + self.opt_setup.currentText() + "'")

            if len(selection) == 0:
                self.show_dialog("No previous training existing for setup " + self.opt_setup.currentText(), "Warning")
            else:
                old_setupID = selection[0][0]
                selection = self.configuration.project.select("SELECT timestamp FROM tbl_standardization WHERE setupID = " + str(old_setupID))
                new_name = self.opt_setup.currentText() + tool_general.string_stripper(selection[0][0], [])
                run = True
                counter = 0
                while run:
                    new_description = new_name + str(counter)
                    selection = self.configuration.project.select("SELECT setupID FROM tbl_standardization_setup WHERE description = '" + new_description + "'")
                    if len(selection) == 0:
                        run = False
                    if counter > 100:
                        raise RuntimeError("Something went wrong with copying an existing training")
                self.configuration.project.insert_copy_standardization(old_setupID, new_description)
                self.show_dialog("The previous training of " + self.opt_setup.currentText() + " were saved as " + new_description + ". Therefore " + self.opt_setup.currentText() + " is now empty. The saved training cannot be edited.", "Information")
                self.update_setup()
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
        path_ext = self.get_file(None, "*.marissadb")
        if not path_ext is None and not path_ext == "":
            self.import_extern(path_ext)
        return

    def import_extern(self, path_ext):
        project_ext = creator_sqlitedb.Inheritance(path_ext)
        try:
            standardization_ext = project_ext.select("SELECT setupID, description FROM tbl_standardization_setup")
        except:
            self.show_dialog("The extern project has not the expected format or is corrupted. his does not work, sorry :(", " Critical")
            return

        if len(standardization_ext) == 0:
            self.show_dialog("The extern project does not contain any standardization.")
        else:
            standardization_IDs = [standardization_ext[i][0] for i in range(len(standardization_ext))]
            standardization_descriptions = [standardization_ext[i][1] for i in range(len(standardization_ext))]

            dialog = dialog_list_choice.GUI(self, description="Choose standardization to import", list=standardization_descriptions, listchoice="single")
            if dialog.exec():
                import_setup_ID = standardization_IDs[standardization_descriptions.index(dialog.result[0])]

                selection = self.configuration.project.select("SELECT 1 FROM tbl_standardization_setup WHERE setupID = " + str(import_setup_ID))

                if len(selection) > 0:
                    answer = self.show_dialog("The chosen setup already exists. Do you want to overwrite the existing training?", "Question")

                    if answer == QtWidgets.QMessageBox.Yes:
                        self.configuration.project.delete_standardization(import_setup_ID)
                    else:
                        return

                self.configuration.project.import_standardization(path_ext, import_setup_ID)
                self.update_setup()
                self.show_dialog("The import of the trained setup " + dialog.result[0] + " was successfull", "Information")
        return

    # tab data
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

    # tab segmentation

    # tab train
    def btn_train_default_reference_clicked(self):
        self.tbl_reference_changed(None)
        return

    def cb_reference_clicked(self, event):
        if self.cb_automatic_reference.isChecked():
            self.tbl_reference.setVisible(False)
            self.btn_train_default_reference.setVisible(False)
        elif self.cb_manual_reference.isChecked():
            self.tbl_reference.setVisible(True)
            self.btn_train_default_reference.setVisible(True)
        return

    def tbl_reference_changed(self, index):
        setupID = str(self.configuration.project.select("SELECT setupID FROM tbl_setup WHERE description = '" + self.opt_setup.currentText() + "'")[0][0])

        if index is None:
            self.reference_selected = np.copy(self.reference_automatic)
        elif index < 0:
            parameterIDs = self.configuration.project.select("SELECT parameterID FROM tbl_match_setup_parameter WHERE setupID = " + setupID + " ORDER BY ordering")
            parameterIDs = [a[0] for a in parameterIDs]
            self.reference_parameter_array = []

            selected = np.unique([self.tbl_data.selectedIndexes()[i].row() for i in range(len(self.tbl_data.selectedIndexes()))]).astype(str).tolist()
            for i in range(self.tbl_data.rowCount()):
                if str(i) in selected:
                    self.reference_parameter_array.append(self.configuration.project.get_data_parameters(self.tbl_data.item(i, 0).text(), parameterIDs))

            self.reference_parameter_array = np.array(self.reference_parameter_array, dtype=str)
            indeces_sort_out = np.unique(np.argwhere(np.array(self.reference_parameter_array).astype(str) == "None")[:,0].flatten())
            self.reference_parameter_array = np.delete(self.reference_parameter_array, indeces_sort_out, axis=0)

            if len(self.reference_parameter_array) < 1:
                self.show_dialog("All chosen data have at least in one of the parameters a None value, hence the setup cannot be trained :( Please use another setup or de-anonymize the parameters necessary for the setup.", "Warning")
                self.tab_update(0)

            self.reference_automatic, counts = np.unique(np.array(['\t'.join(row) for row in self.reference_parameter_array]), return_counts=True)
            idx = np.argmax(counts)
            self.reference_selected = np.array(self.reference_automatic[idx].tolist().split("\t"))
            self.reference_automatic = np.array(self.reference_automatic[idx].tolist().split("\t"))
        else:
            self.reference_selected = [None for _ in range(len(self.reference_selected))]
            for i in range(self.tbl_reference.selectedIndexes()[0].row()+1):
                self.reference_selected[i] = self.tbl_reference.cellWidget(i,1).currentText()

        columns = ["parameter", "reference value"]
        rows = []
        #selection = self.configuration.project.select("SELECT description, VM FROM tbl_parameter WHERE parameterID IN (SELECT parameterID FROM tbl_match_setup_parameter WHERE setupID = " + setupID + " ORDER BY ordering)")
        selection = self.configuration.project.select("SELECT p.description, p.VM FROM tbl_parameter AS p INNER JOIN tbl_match_setup_parameter AS msp ON msp.parameterID = p.parameterID WHERE msp.setupID = " + setupID + " ORDER BY msp.ordering")

        for i in range(len(selection)):
            if selection[i][1] > 1:
                for j in range(selection[i][1]):
                    rows.append(selection[i][0] + " " + str(j+1))
            else:
                rows.append(selection[i][0])

        self.tbl_reference.clear()
        self.tbl_reference.setRowCount(0)

        self.tbl_reference.setRowCount(len(rows))
        self.tbl_reference.setColumnCount(len(columns))
        self.tbl_reference.setHorizontalHeaderLabels(columns)

        for i in range(len(rows)):
            filter = []
            for j in range(i):
                filter.append("self.reference_parameter_array[:," + str(j) + "] == \"" + self.tbl_reference.cellWidget(j,1).currentText() + "\"")

            if len(filter) == 0:
                options = np.unique(self.reference_parameter_array[:,i])
            else:
                if len(filter) == 1:
                    filter = filter[0]
                else:
                    filter = "np.logical_and(" + ", np.logical_and(".join([filter[a] for a in range(len(filter)-1)]) + ", " + filter[1] + "".join([")"] * (len(filter)-1))

                options = np.unique(self.reference_parameter_array[eval(filter),i])

            self.tbl_reference.setItem(i, 0, QtWidgets.QTableWidgetItem(rows[i]))
            tbl_cb = QtWidgets.QComboBox(self.tbl_reference)
            tbl_cb.resize(200, 20)
            tbl_cb.setFont(QtGui.QFont("Arial", 10))
            tbl_cb.addItems(options.tolist())
            if not self.reference_selected[i] is None:
                tbl_cb.setCurrentText(self.reference_selected[i])
            tbl_cb.currentIndexChanged.connect(self.tbl_reference_changed)
            self.tbl_reference.setCellWidget(i, 1, tbl_cb)
        return

    def btn_icon_new_train_clicked(self):
        setupID = str(self.configuration.project.select("SELECT setupID FROM tbl_setup WHERE description = '" + self.opt_setup.currentText() + "'")[0][0])
        if self.cb_manual_reference.isChecked():
            reference = [self.tbl_reference.cellWidget(i,1).currentText() for i in range(len(self.reference_automatic))]
        else:
            reference = None
        self.configuration.project.insert_standardization(setupID, reference)
        self.show_dialog("Training for " + self.opt_setup.currentText() + " successfully performed.", "Information")
        self.tab_update(0)
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
