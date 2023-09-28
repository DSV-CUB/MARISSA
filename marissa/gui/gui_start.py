import sys
import os
import numpy as np
import shutil
from PyQt5 import QtWidgets, QtCore

from marissa.gui import dialog_info, dialog_add_edit_project, gui_project
from marissa.toolbox.creators import creator_gui
from marissa.modules.database import marissadb


class GUI(creator_gui.Inheritance):
    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.btn_icon_logo_info.clicked.connect(self.btn_icon_logo_info_clicked)
        self.btn_project_start.clicked.connect(self.btn_project_start_clicked)
        self.btn_icon_new_project.clicked.connect(self.btn_icon_new_project_clicked)
        self.btn_icon_edit_project.clicked.connect(self.btn_icon_edit_project_clicked)
        self.btn_icon_delete_project.clicked.connect(self.btn_icon_delete_project_clicked)
        self.btn_icon_import_project.clicked.connect(self.btn_icon_import_project_clicked)
        self.btn_icon_export_project.clicked.connect(self.btn_icon_export_project_clicked)
        
        self.projects = []
        self.update_projects()
        return

    # Override closeEvent
    def closeEvent(self, event):
        event.accept()
        return

    def update_projects(self):
        self.opt_project.clear()
        self.projects = []
        for root, _, files in os.walk(self.configuration.path_projects):
            for file in files:
                if file.endswith(".marissadb"):
                    self.projects.append([file.replace(".marissadb", ""), os.path.join(root, file)])

        self.projects = np.array(self.projects)

        if len(self.projects) > 0:
            self.opt_project.addItems(self.projects[:,0].flatten().tolist())

            try:
                self.opt_project.setCurrentText(self.projects[np.argwhere(self.projects[:,1] == self.configuration.path_project).flatten()[0], 0])
            except:
                pass
        self.configuration.path_project = None
        self.configuration.project = None
        return

    def btn_project_start_clicked(self):
        if len(self.projects) > 0:
            self.configuration.path_project = self.projects[np.argwhere(self.projects[:,0] == self.opt_project.currentText()).flatten()[0], 1]
            self.configuration.project = marissadb.Module(self.configuration.path_project)

            global gui_run
            gui_run = gui_project.GUI(None, self.configuration)
            gui_run.show()
            self.close()
        return

    def btn_icon_delete_project_clicked(self):
        if len(self.projects) > 0:
            self.configuration.path_project = None
            self.configuration.project = None

            if self.show_dialog("Are you sure to delete the project '" + self.opt_project.currentText() + "'", "Question") == QtWidgets.QMessageBox.Yes:
                try:
                    os.remove(self.projects[np.argwhere(self.projects[:,0] == self.opt_project.currentText()).flatten()[0], 1])
                    self.show_dialog("The project '" + self.opt_project.currentText() + "' was successfully deleted.", "Information")
                except:
                    self.show_dialog("The project could not be deleted, please try again later.", "Critical")
                self.update_projects()
        return

    def btn_icon_new_project_clicked(self):
        self.configuration.path_project = None
        self.configuration.project = None

        dialog = dialog_add_edit_project.GUI(self, self.configuration)
        if dialog.exec():
            global gui_run
            gui_run = gui_project.GUI(None, self.configuration)
            gui_run.show()
            self.close()
        return

    def btn_icon_edit_project_clicked(self):
        if len(self.projects) > 0:
            self.configuration.path_project = self.projects[np.argwhere(self.projects[:,0] == self.opt_project.currentText()).flatten()[0], 1]
            self.configuration.project = marissadb.Module(self.configuration.path_project)

            dialog = dialog_add_edit_project.GUI(self, self.configuration)
            if dialog.exec():
                global gui_run
                gui_run = gui_project.GUI(None, self.configuration)
                gui_run.show()
                self.close()
        return

    def btn_icon_import_project_clicked(self):
        path_from = self.get_file(None, "MARISSA project (*.marissadb)")
        if path_from is None or path_from == "":
            pass
        else:
            try:
                db = marissadb.Module(path_from)
                name = db.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0]
                path_to = os.path.join(self.configuration.path_projects, name + ".marissadb")

                if os.path.isfile(path_to):
                    self.show_dialog("A project with the same name '" + name + "' already exists. Overwriting is not allowed. Please rename or delete the existing project.")
                else:
                    shutil.copyfile(path_from, path_to)
                    self.update_projects()
                    self.opt_project.setCurrentText(name)
                    self.show_dialog("Project import successfull.", "Information")
            except:
                self.show_dialog("The project could not be imported as the file is potentially corrupted.", "Critical")
        return

    def btn_icon_export_project_clicked(self):
        from marissa.gui import dialog_list_choice

        dialog = dialog_list_choice.GUI(self, self.configuration, description="How do you want to export the project?", list=["with DICOM data", "without DICOM data"])
        if dialog.exec():
            include_data = True if dialog.result[0] == "with DICOM data" else False

            path_to = self.get_directory(None)
            if not path_to is None and path_to != "":
                try:
                    path_from = self.projects[np.argwhere(self.projects[:,0] == self.opt_project.currentText()).flatten()[0], 1]
                    path_to = os.path.join(path_to, os.path.basename(path_from))if include_data else os.path.join(path_to, os.path.basename(path_from)).replace(".marissadb", "_NODATA.marissadb")
                    shutil.copyfile(path_from, path_to)
                    if not include_data:
                        db = marissadb.Module(path_to)
                        db.delete_data_all()
                        db.execute("UPDATE tbl_info SET parameter='" + self.opt_project.currentText() + "_NODATA' WHERE ID='name'")
                        db.vacuum()

                    self.show_dialog("Project export successfull.", "Information")
                except:
                    self.show_dialog("Project export failed.", "Critical")
        return

    def btn_icon_logo_info_clicked(self):
        dialog = dialog_info.GUI(self, self.configuration)
        dialog.exec()
        return

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())