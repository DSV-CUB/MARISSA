import sys
import os
import numpy as np
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_dialog
from marissa.toolbox.tools import tool_general, tool_pydicom

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None, tab="import"):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.btn_icon_import_segmentation.clicked.connect(self.btn_icon_import_segmentation_clicked)

        self.btn_icon_new_rule.clicked.connect(self.btn_icon_new_rule_clicked)
        self.btn_icon_delete_rule.clicked.connect(self.btn_icon_delete_rule_clicked)
        self.btn_icon_new_segmentation_rule.clicked.connect(self.btn_icon_new_segmentation_rule_clicked)

        self.worked = False
        self.rules = []

        self.fill()

        index = [self.tabs_segmentation.widget(i).objectName() for i in range(self.tabs_segmentation.count())].index("tab_" + tab)
        self.tabs_segmentation.setCurrentIndex(index)
        return

    def closeEvent(self, event):
        if self.worked:
            self.accept()
        event.accept()
        return

    def fill(self):
        # tab RULE
        selection = self.configuration.project.select("SELECT DISTINCT description, creator FROM tbl_segmentation")
        segmentations = [selection[i][0] + " | " + selection[i][1] for i in range(len(selection))]

        self.lst_segmentation_1.clear()
        self.lst_segmentation_2.clear()
        self.lst_segmentation_1.addItems(segmentations)
        self.lst_segmentation_2.addItems(segmentations)
        try:
            self.lst_segmentation_1.item(0).setSelected(True)
            self.lst_segmentation_2.item(0).setSelected(True)
        except:
            pass

        metrics = tool_general.get_metric_from_masks()
        self.lst_metric.addItems(metrics)
        return

    # Tab IMPORT
    def btn_icon_import_segmentation_clicked(self):
        description = tool_general.string_stripper(self.txt_description_import.text())
        creator = tool_general.string_stripper(self.txt_creator_import.text())

        if description == "":
            self.show_dialog("Please provide a description" "Critical")
        elif creator == "":
            self.show_dialog("Please provide a creator", "Critical")
        else:
            path = self.get_directory(None)
            counter = self.configuration.project.import_segmentation(path, description, creator, ("npy" if self.rb_npy.isChecked() else "pickle"), False)
            self.worked = True
            answer = self.show_dialog(str(counter) + " segmentations were imported. Do you want to add further segmentations?", "Question")
            if answer == QtWidgets.QMessageBox.No:
                self.accept()
        return

    # Tab Rule
    def update_rules(self):
        self.lst_rule.clear()
        for i in range(len(self.rules)):
            rule = " ".join(self.rules[i])
            self.lst_rule.addItem(rule)
        return

    def btn_icon_new_rule_clicked(self):
        try:
            metric = self.lst_metric.currentItem().text()
            operator = self.opt_operator.currentText()
            value = str(self.sb_value.value())
            self.rules.append([metric, operator, value])
            self.update_rules()
        except:
            self.show_dialog("Please choose a metric.", "Critical")
        return

    def btn_icon_delete_rule_clicked(self):
        try:
            index = self.lst_rule.currentRow()
            del self.rules[index]
            self.update_rules()
        except:
            pass
        return

    def btn_icon_new_segmentation_rule_clicked(self):
        try:
            description1, creator1 = tuple(self.lst_segmentation_1.selectedItems()[0].text().split(" | "))
            description2, creator2 = tuple(self.lst_segmentation_2.selectedItems()[0].text().split(" | "))

            description = self.txt_description_rule.text()
            creator = self.txt_creator_rule.text()
            counter = 0

            if tool_general.string_stripper(description, []) == "":
                self.show_dialog("Please provide a description.", "Critical")
            if tool_general.string_stripper(creator, []) == "":
                self.show_dialog("Please provide a creator.", "Critical")

            selection = self.configuration.project.select("SELECT s1.SOPinstanceUID, s1.mask, s2.mask FROM (tbl_segmentation AS s1 INNER JOIN tbl_segmentation AS s2 ON s1.SOPinstanceUID = s2.SOPinstanceUID) WHERE s1.description = '" + description1 + "' AND s1.creator = '" + creator1 + "' AND s2.description = '" + description2 + "' AND s2.creator = '" + creator2 + "'")

            for i in range(len(selection)):
                SOPinstanceUID = selection[i][0]
                mask1 = selection[i][1]
                mask2 = selection[i][2]

                dcm = self.configuration.project.get_data(SOPinstanceUID)[0]
                array = tool_pydicom.get_dcm_pixel_data(dcm, rescale=True, representation=False)
                try:
                    ps = np.array(dcm.PixelSpacing, dtype=float)
                except:
                    ps = [1, 1]

                if self.opt_connector.currentText().endswith("NOT"):
                    mask2 = np.abs(mask2-1)

                combined = mask1 + mask2

                if self.opt_connector.currentText().startswith("AND"):
                    combined[combined<2] = 0
                    combined[combined>=2] = 1

                if self.opt_connector.currentText().startswith("OR"):
                    combined[combined>=1] = 1

                if self.opt_connector.currentText().startswith("XOR"):
                    combined[combined>=2] = 0

                rule_pass = True
                for rule in self.rules:
                    metric_value = tool_general.get_metric_from_masks(mask1, mask2, rule[0], values=array, voxel_sizes=ps)
                    rule_pass = rule_pass and eval("metric_value " + rule[1] + " " + rule[2])

                if rule_pass:
                    worked = self.configuration.project.insert_segmentation(SOPinstanceUID, description, creator, tool_general.mask2contour(combined), combined, len(np.shape(combined)))

                    if worked:
                        counter = counter + 1

            self.worked = True
            answer = self.show_dialog(str(counter) + " segmentations were created. Do you want to add further segmentations?", "Question")
            if answer == QtWidgets.QMessageBox.No:
                self.accept()
        except:
            pass
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
