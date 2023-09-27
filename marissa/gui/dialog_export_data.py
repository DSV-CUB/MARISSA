import sys
import os
import datetime
import numpy as np
import pickle
from PyQt5 import QtWidgets, QtCore

from marissa.toolbox.tools import tool_general, tool_hadler
from marissa.toolbox.creators import creator_dialog

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.cb_contours.stateChanged.connect(self.cb_contours_clicked)
        self.opt_numpy.toggled.connect(self.opt_numpy_clicked)
        self.opt_pickle.toggled.connect(self.opt_pickle_clicked)
        self.btn_select.clicked.connect(self.btn_select_clicked)
        self.btn_deselect.clicked.connect(self.btn_deselect_clicked)
        self.btn_icon_export.clicked.connect(self.btn_icon_export_clicked)

        selection = self.configuration.project.select("SELECT DISTINCT description, creator FROM tbl_segmentation ORDER BY description")
        for i in range(len(selection)):
            self.lst_contours.addItem(selection[i][0] + " | " + selection[i][1])
        return

    def closeEvent(self, event):
        event.accept()
        return

    def cb_contours_clicked(self, state):
        if state == QtCore.Qt.Checked:
            self.lbl_hide.hide()
        else:
            self.lbl_hide.show()
        return

    def opt_numpy_clicked(self):
        if self.opt_numpy.isChecked():
            self.opt_pickle.setChecked(False)
        return

    def opt_pickle_clicked(self):
        if self.opt_pickle.isChecked():
            self.opt_numpy.setChecked(False)
        return

    def btn_select_clicked(self):
        self.lst_contours.selectAll()
        self.lst_contours.model().layoutChanged.emit()
        return

    def btn_deselect_clicked(self):
        self.lst_contours.clearSelection()
        self.lst_contours.model().layoutChanged.emit()
        return

    def btn_icon_export_clicked(self):
        if not self.cb_contours.isChecked() and not self.cb_data.isChecked():
            self.show_dialog("Nothing to export.", "Information")
        else:
            path_out = self.get_directory(None)
            if path_out is None or path_out == "":
                pass
            else:
                path_out = os.path.join(path_out, "marissa_export_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

                if self.cb_data.isChecked():
                    selection = self.configuration.project.select("SELECT SOPinstanceUID FROM tbl_data")
                    for i in range(len(selection)):
                        dcm = self.configuration.project.get_data(selection[i][0])[0]
                        tool_general.save_dcm(dcm, os.path.join(path_out, "data"), True)

                if self.cb_contours.isChecked():
                    segmentations = [item.text().split(" | ") for item in self.lst_contours.selectedItems()]
                    for segmentation in segmentations:
                        selection = self.configuration.project.select("SELECT SOPinstanceUID, mask FROM tbl_segmentation WHERE description = '" + segmentation[0] + "' and creator = '" + segmentation[1] + "'")
                        for i in range(len(selection)):
                            mask = selection[i][1]

                            filedir = os.path.join(path_out, "segmentation", segmentation[0] + "_" + segmentation[1])

                            if self.opt_numpy.isChecked():
                                os.makedirs(filedir, exist_ok=True)
                                file = open(os.path.join(filedir, selection[i][0] + ".npy"), "wb")
                                np.save(file, mask)
                                file.close()
                            elif self.opt_pickle.isChecked():
                                dcm = self.configuration.project.get_data(selection[i][0])[0]
                                #annotation = tool_hadler.mask_to_LL_annotation(tool_general.mask_highres(mask), dcm)
                                annotation = tool_hadler.mask_to_LL_annotation(mask, dcm)

                                os.makedirs(os.path.join(filedir, str(dcm[0x0020, 0x000D].value)), exist_ok=True)
                                file = open(os.path.join(filedir, str(dcm[0x0020, 0x000D].value), selection[i][0] + ".pickle"), "wb")
                                pickle.dump(annotation, file)
                                file.close()

                self.show_dialog("Export successfully done.", "Information")
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
