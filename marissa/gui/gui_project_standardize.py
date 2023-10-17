import sys
import os
import numpy as np
import pydicom
import pickle
import datetime
from PyQt5 import QtWidgets, QtGui, QtCore

from marissa.toolbox.creators import creator_gui
from marissa.toolbox.tools import tool_general, tool_pydicom, tool_hadler, tool_excel
from marissa.gui import gui_project

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))
        self.setWindowTitle("MARISSA - Project Standardize : " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])

        # EVENTS
        self.rb_extern.toggled.connect(self.rb_extern_clicked)
        self.rb_intern.toggled.connect(self.rb_intern_clicked)
        self.rb_numpy.toggled.connect(self.rb_numpy_clicked)
        self.rb_pickle.toggled.connect(self.rb_pickle_clicked)
        self.lbl_hide_external.setVisible(False)

        self.txt_path_dicom.mousePressEvent = self.txt_path_dicom_clicked
        self.txt_path_segmentation.mousePressEvent = self.txt_path_segmentation_clicked

        self.btn_data_sql.clicked.connect(self.btn_data_sql_clicked)

        self.btn_icon_new_standardize.clicked.connect(self.btn_icon_new_standardize_clicked)

        # FILL
        selection = self.configuration.project.select("SELECT DISTINCT description, creator FROM tbl_segmentation ORDER BY description")
        segmentations = [selection[i][0] + " | " + selection[i][1] for i in range(len(selection))]
        self.opt_segmentation.addItems(segmentations)

        selection = self.configuration.project.select("SELECT DISTINCT ss.description FROM tbl_standardization_setup AS ss INNER JOIN tbl_standardization AS s ON ss.setupID = s.setupID ORDER BY description")
        standardized_setups = [selection[i][0] for i in range(len(selection))]
        self.opt_setup.addItems(standardized_setups)

        # RUN
        self.btn_data_sql_clicked()
        return

    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return

    def rb_extern_clicked(self):
        if self.rb_extern.isChecked():
            self.rb_intern.setChecked(False)
            self.lbl_hide_external.setVisible(False)
            self.lbl_hide_internal.setVisible(True)
        else:
            self.rb_intern.setChecked(True)
        return

    def rb_intern_clicked(self):
        if self.rb_intern.isChecked():
            self.rb_extern.setChecked(False)
            self.lbl_hide_external.setVisible(True)
            self.lbl_hide_internal.setVisible(False)
        else:
            self.rb_extern.setChecked(True)
        return

    def rb_numpy_clicked(self):
        if self.rb_numpy.isChecked():
            self.rb_pickle.setChecked(False)
        else:
            self.rb_pickle.setChecked(True)
        return

    def rb_pickle_clicked(self):
        if self.rb_pickle.isChecked():
            self.rb_numpy.setChecked(False)
        else:
            self.rb_numpy.setChecked(True)
        return

    def txt_path_dicom_clicked(self, event):
        self.get_directory(self.txt_path_dicom)
        return

    def txt_path_segmentation_clicked(self, event):
        self.get_directory(self.txt_path_segmentation)
        return

    def btn_data_sql_clicked(self):
        where = self.txt_data_where.toPlainText().strip()
        if not where == "" and not where.upper().startswith("WHERE") and not where.upper().startswith("LIMIT") and not where.upper().startswith("ORDER BY") and not where.upper().startswith("GROUP BY"):
            where = "WHERE " + where

        try:
            columns = ["SOPInstanceUID", "StudyinstanceUID", "seriesnumber", "instancenumber", "seriesdescription", "identifier", "age", "gender", "size", "weight", "description", "acquisitiondatetime", "timestamp"]
            selection = self.configuration.project.select("SELECT " + ",".join(columns) + " FROM tbl_data " + where)
        except:
            selection = []

        if len(selection) == 0:
            self.show_dialog("No data found or there is a misstake in the SQL.", "Critical")
        else:
            self.tbl_data.clear()
            self.tbl_data.setRowCount(0)
            self.tbl_data.setRowCount(len(selection))
            self.tbl_data.setColumnCount(len(columns))
            self.tbl_data.setHorizontalHeaderLabels(columns)
            for i in range(len(selection)):
                for j in range(len(selection[i])):
                    self.tbl_data.setItem(i, j, QtWidgets.QTableWidgetItem(str(selection[i][j])))
            self.tbl_data.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        return

    def btn_icon_new_standardize_clicked(self):
        import numpy as np
        from matplotlib import pyplot as plt

        dcms = []
        masks = []

        setupID = self.configuration.project.select("SELECT setupID FROM tbl_standardization_setup WHERE description = '" + self.opt_setup.currentText() + "'")[0][0]
        skip_unknown = self.cb_skip_unknown.isChecked()

        if self.rb_intern.isChecked():
            where = self.txt_data_where.toPlainText().strip()
            if not where == "" and not where.upper().startswith("WHERE") and not where.upper().startswith("LIMIT") and not where.upper().startswith("ORDER BY") and not where.upper().startswith("GROUP BY"):
                where = "WHERE " + where

            description, creator = self.opt_segmentation.currentText().split(" | ")
            selection = self.configuration.project.select("SELECT SOPinstanceUID, mask FROM tbl_segmentation WHERE SOPinstanceUID IN (SELECT SOPinstanceUID FROM tbl_data " + where + ") AND description = '" + description + "' AND creator = '" + creator + "'")
            dcms = [self.configuration.project.get_data(selection[i][0])[0] for i in range(len(selection))]
            masks = [selection[i][1] for i in range(len(selection))]

        elif self.rb_extern.isChecked():
            path_dcm = self.txt_path_dicom.text()
            path_segm = self.txt_path_segmentation.text()
            filetype_segm = ("numpy" if self.rb_numpy.isChecked() else "pickle" if self.rb_pickle.isChecked() else "")

            # load dicoms
            ids_dcm = []
            dcms_read = []
            for root, _, files in os.walk(path_dcm):
                for file in files:
                    try:
                        dcm = pydicom.dcmread(os.path.join(root, file))
                        SUID = str(dcm[0x0008, 0x0018].value)
                        ids_dcm.append(SUID)
                        dcms_read.append(dcm)
                    except:
                        pass
            ids_dcm = np.array(ids_dcm)

            # load segmentations
            ids_mask = []
            masks_read = []
            for root, _, files in os.walk(path_segm):
                for file in files:
                    if filetype_segm == "numpy" and file.endswith(".npy"):
                        SUID = file.replace(".npy", "")
                        file = open(os.path.join(root, file), "rb")
                        mask = np.load(file, allow_pickle=True)
                        file.close()
                    elif filetype_segm == "pickle" and file.endswith(".pickle"):
                        SUID = file.replace(".pickle", "")
                        file = open(os.path.join(root, file), "rb")
                        data = pickle.load(file)
                        file.close()
                        mask = tool_hadler.from_polygon(data["lv_myo"]["cont"], data["lv_myo"]["imageSize"])
                    else:
                        continue

                    ids_mask.append(SUID)
                    masks_read.append(mask)

            # match data and segmentation
            for i in range(len(ids_mask)):
                try:
                    index = np.argwhere(ids_dcm==ids_mask[i]).flatten()[0]
                    dcms.append(dcms_read[index])
                    masks.append(masks_read[i])
                except:
                    pass

        if len(dcms) != len(masks):
            self.show_dialog("There is an implementation failure as the number of DICOM data do not fit the number of segmentation masks", "Critical")
        elif len(dcms) == 0:
            self.show_dialog("No data to standardize.", "Information")
        else:
            path_out = self.get_directory(None)

            if path_out is None or path_out == "":
                return

            path_out = os.path.join(path_out, "marissa_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            os.makedirs(path_out, exist_ok="true")

            meanvalues = []

            for i in range(len(dcms)):
                dcm = dcms[i]
                mask = masks[i]

                marissadata = self.configuration.project.get_standardization(dcm, mask, setupID, skip_unknown=skip_unknown)
                case_path = tool_general.save_dcm(dcm, path_out)
                md_path = marissadata.save(case_path)

                xlsx = tool_excel.Setup(md_path.replace(".marissadata", ".xlsx"))
                xlsx.add_worksheet("MARISSA")
                xlsx.write_line("MARISSA", ["SOPinstanceUID", str(dcm[0x0008, 0x0018].value)])
                xlsx.write_line("MARISSA", ["MARISSA Version", str(marissadata.version).replace(",",".")])
                xlsx.write_line("MARISSA", ["MARISSA Project", str(self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])])
                xlsx.write_line("MARISSA", ["Standardization", self.configuration.project.select("SELECT description FROM tbl_standardization_setup WHERE setupID=" + str(setupID))[0][0], "(" + ", ".join(self.configuration.project.select("SELECT regressiontype, mode, ytype, clustertype, CAST(bins AS TEXT) FROM tbl_standardization_setup WHERE setupID=" + str(setupID))[0]) + ")"])
                xlsx.write_line("MARISSA", ["timestamp", str(marissadata.creation)])
                xlsx.write_line("MARISSA", "")
                xlsx.write_line("MARISSA", ["CP", "CP value", "mean", "std", "", "single values"])
                xlsx.write_line("MARISSA", ["Original", "", "{:.2f}".format(np.mean(marissadata.value_progression[0])), "{:.2f}".format(np.std(marissadata.value_progression[0])), ""] + np.round(marissadata.value_progression[0], 2).astype(str).tolist())
                for i in range(len(marissadata.value_progression)-1):
                    if len(marissadata.parameters) < len(marissadata.parameters_values):
                        xlsx.write_line("MARISSA", [marissadata.parameters[i], [" & ".join([str(m) for m in marissadata.parameters_values])], "{:.2f}".format(np.mean(marissadata.value_progression[i+1])), "{:.2f}".format(np.std(marissadata.value_progression[i+1])), ""] + np.round(marissadata.value_progression[i+1], 2).astype(str).tolist())
                    else:
                        xlsx.write_line("MARISSA", [marissadata.parameters[i], marissadata.parameters_values[i], "{:.2f}".format(np.mean(marissadata.value_progression[i+1])), "{:.2f}".format(np.std(marissadata.value_progression[i+1])), ""] + np.round(marissadata.value_progression[i+1], 2).astype(str).tolist())
                #xlsx.write_line("MARISSA", ["Standardized", "", "{:.2f}".format(np.mean(marissadata.value_progression[-1])), "{:.2f}".format(np.std(marissadata.value_progression[-1])), ""] + np.round(marissadata.value_progression[-1], 2).astype(str).tolist())
                xlsx.save()

                meanvalues.append([np.mean(marissadata.value_progression[j]) for j in range(len(marissadata.value_progression))])

            meanvalues = np.array(meanvalues)

            fig, ax = plt.subplots(1,1, figsize=(10, 10))


            ax.scatter([0] * np.shape(meanvalues)[0], meanvalues[:,0], c="#F4B183", s=10, zorder=10)
            for i in range(1, np.shape(meanvalues)[1]):
                ax.scatter(np.array([i] * np.shape(meanvalues)[0]) - 0.5, meanvalues[:,i], c="#9DC3E6", s=10, zorder=10)
            ax.scatter(np.array([i+1] * np.shape(meanvalues)[0])-1, meanvalues[:,-1], c="#A9D18E", s=10, zorder=10)


            for i in range(len(meanvalues)):
                ax.plot([0] + np.arange(0.5, np.shape(meanvalues)[1]-1, 1).tolist() + [np.shape(meanvalues)[1]-1], meanvalues[i,:].tolist() + [meanvalues[i,-1]], color="gray", linestyle="--", linewidth=1)

            ax.set_xlim([-0.5, np.shape(meanvalues)[1] - 0.5])

            ax.set_xticks([0] + np.arange(0.5, np.shape(meanvalues)[1]-0.5, 1).tolist() + [np.shape(meanvalues)[1]-1])
            ax.set_xticklabels(["original"] + marissadata.parameters + ["standardized"], rotation=45, ha='right')

            ax.set_ylabel("mean value")
            ax.set_xlabel("standardization progression")
            ax.grid(axis="y")

            title = "Data Standardization of"
            title = title + " " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'quantitative'")[0][0]
            title = title + " in a " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'organ'")[0][0]

            selection = self.configuration.project.select("SELECT description, bins FROM tbl_standardization_setup WHERE setupID = " + str(setupID))
            title = title + "\n" "with the " + selection[0][0] + " setup having " + str(selection[0][1]) + " bins"

            ax.set_title(title)

            plt.tight_layout()
            plt.savefig(os.path.join(path_out, "standardization_progress.jpg"), dpi=300)

            self.show_dialog("Standardization successfully done. :)", "Information")
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
