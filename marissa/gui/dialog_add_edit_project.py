import sys
import os
import datetime
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_dialog
from marissa.toolbox.tools import tool_general
from marissa.modules.database import marissadb


class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.btn_icon_new_subject.clicked.connect(self.btn_icon_new_subject_clicked)
        self.btn_icon_delete_subject.clicked.connect(self.btn_icon_delete_subject_clicked)
        self.btn_icon_new_organ.clicked.connect(self.btn_icon_new_organ_clicked)
        self.btn_icon_delete_organ.clicked.connect(self.btn_icon_delete_organ_clicked)
        self.btn_icon_new_quantitative.clicked.connect(self.btn_icon_new_quantitative_clicked)
        self.btn_icon_delete_quantitative.clicked.connect(self.btn_icon_delete_quantitative_clicked)
        self.btn_icon_new.clicked.connect(self.btn_icon_new_clicked)
        self.btn_icon_edit.clicked.connect(self.btn_icon_edit_clicked)

        self.update_comboboxes()

        if self.configuration.path_project is None:
            self.btn_icon_new.setVisible(True)
            self.btn_icon_edit.setVisible(False)
        else:
            self.btn_icon_new.setVisible(False)
            self.btn_icon_edit.setVisible(True)

            name = self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0]
            self.setWindowTitle("MARISSA - Edit Project : " + name)
            self.txt_name.setText(name)
            self.opt_subject.setCurrentText(self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'subject'")[0][0])
            self.opt_organ.setCurrentText(self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'organ'")[0][0])
            self.opt_quantitative.setCurrentText(self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'quantitative'")[0][0])
        return

    def closeEvent(self, event):
        event.accept()
        return

    def update_comboboxes(self):
        self.update_opt_subject()
        self.update_opt_organ()
        self.update_opt_quantitative()
        return

    def update_opt_subject(self):
        self.opt_subject.clear()
        selection = self.configuration.database.select("SELECT parameter FROM tbl_project WHERE ID = 'subject' ORDER BY parameter ASC;")
        for i in selection:
            self.opt_subject.addItem(i[0])
        return

    def btn_icon_new_subject_clicked(self):
        answer = self.show_dialog("Add subject", "Input")
        if answer and not answer is None and not answer == "":
            answer = tool_general.string_stripper(answer)
            self.configuration.database.execute("INSERT OR IGNORE INTO tbl_project VALUES ('subject', '" + answer + "')")
            self.update_opt_subject()
            self.opt_subject.setCurrentText(answer)
        return

    def btn_icon_delete_subject_clicked(self):
        if not self.opt_subject.currentText() == "" and self.show_dialog("Are you sure to delete the subject '" + self.opt_subject.currentText() + "' from the list?", "Question") == QtWidgets.QMessageBox.Yes:
            self.configuration.database.execute("DELETE FROM tbl_project WHERE ID = 'subject' AND parameter = '" + self.opt_subject.currentText() + "'")
            self.update_opt_subject()
        return

    def update_opt_organ(self):
        self.opt_organ.clear()
        selection = self.configuration.database.select("SELECT parameter FROM tbl_project WHERE ID = 'organ' ORDER BY parameter ASC;")
        for i in selection:
            self.opt_organ.addItem(i[0])
        return

    def btn_icon_new_organ_clicked(self):
        answer = self.show_dialog("Add organ", "Input")
        if answer and not answer is None and not answer == "":
            answer = tool_general.string_stripper(answer)
            self.configuration.database.execute("INSERT OR IGNORE INTO tbl_project VALUES ('organ', '" + answer + "')")
            self.update_opt_organ()
            self.opt_organ.setCurrentText(answer)
        return

    def btn_icon_delete_organ_clicked(self):
        if not self.opt_organ.currentText() == "" and self.show_dialog("Are you sure to delete the organ '" + self.opt_organ.currentText() + "' from the list?", "Question") == QtWidgets.QMessageBox.Yes:
            self.configuration.database.execute("DELETE FROM tbl_project WHERE ID = 'organ' AND parameter = '" + self.opt_organ.currentText() + "'")
            self.update_opt_organ()
        return

    def update_opt_quantitative(self):
        self.opt_quantitative.clear()
        selection = self.configuration.database.select("SELECT parameter FROM tbl_project WHERE ID = 'quantitative' ORDER BY parameter ASC;")
        for i in selection:
            self.opt_quantitative.addItem(i[0])
        return

    def btn_icon_new_quantitative_clicked(self):
        answer = self.show_dialog("Add quantitative", "Input")
        if answer and not answer is None and not answer == "":
            answer = tool_general.string_stripper(answer)
            self.configuration.database.execute("INSERT OR IGNORE INTO tbl_project VALUES ('quantitative', '" + answer + "')")
            self.update_opt_quantitative()
            self.opt_quantitative.setCurrentText(answer)
        return

    def btn_icon_delete_quantitative_clicked(self):
        if not self.opt_quantitative.currentText() == "" and self.show_dialog("Are you sure to delete the quantitative '" + self.opt_quantitative.currentText() + "' from the list?", "Question") == QtWidgets.QMessageBox.Yes:
            self.configuration.database.execute("DELETE FROM tbl_project WHERE ID = 'quantitative' AND parameter = '" + self.opt_quantitative.currentText() + "'")
            self.update_opt_quantitative()
        return

    def btn_icon_new_clicked(self):
        if self.txt_name.text() == "":
            self.show_dialog("Please provide a project name.", "Critical")
            return
        project_name = tool_general.string_stripper(self.txt_name.text())

        if self.opt_subject.currentText() == "":
            self.show_dialog("Please provide a subject.", "Critical")
            return

        if self.opt_organ.currentText() == "":
            self.show_dialog("Please provide an organ.", "Critical")
            return

        if self.opt_quantitative.currentText() == "":
            self.show_dialog("Please provide a quantitative.", "Critical")
            return

        self.configuration.path_project = os.path.join(self.configuration.path_projects, project_name + ".marissadb")

        if os.path.isfile(self.configuration.path_project):
            self.show_dialog("A project with the name '" + project_name + "' already exists. Creation aborted.", "Critical")
            self.configuration.path_project = None
            return

        self.configuration.project = marissadb.Module(self.configuration.path_project)
        self.configuration.project.execute("INSERT INTO tbl_info VALUES('version', '" + self.configuration.database.select("SELECT parameter FROM tbl_info WHERE ID = 'version'")[0][0] + "')")
        self.configuration.project.execute("INSERT INTO tbl_info VALUES('name', '" + project_name + "')")
        self.configuration.project.execute("INSERT INTO tbl_info VALUES('subject', '" + self.opt_subject.currentText() + "')")
        self.configuration.project.execute("INSERT INTO tbl_info VALUES('organ', '" + self.opt_organ.currentText() + "')")
        self.configuration.project.execute("INSERT INTO tbl_info VALUES('quantitative', '" + self.opt_quantitative.currentText() + "')")
        self.configuration.project.execute("INSERT INTO tbl_info VALUES('timestamp', '" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "')")

        self.accept()
        return

    def btn_icon_edit_clicked(self):
        if self.txt_name.text() == "":
            self.show_dialog("Please provide a project name.", "Critical")
            return
        project_name = tool_general.string_stripper(self.txt_name.text())

        if self.opt_subject.currentText() == "":
            self.show_dialog("Please provide a subject.", "Critical")
            return

        if self.opt_organ.currentText() == "":
            self.show_dialog("Please provide an organ.", "Critical")
            return

        if self.opt_quantitative.currentText() == "":
            self.show_dialog("Please provide a quantitative.", "Critical")
            return

        new_path = os.path.join(self.configuration.path_projects, project_name + ".marissadb")

        if os.path.isfile(new_path) and new_path != self.configuration.path_project:
            self.show_dialog("Another project with the name '" + project_name + "' already exists. Creation aborted.", "Critical")
            return
        elif new_path != self.configuration.path_project:
            os.rename(self.configuration.path_project, new_path)
            self.configuration.path_project = new_path
            self.configuration.project = marissadb.Module(self.configuration.path_project)

        self.configuration.project.execute("UPDATE tbl_info SET parameter = '" + project_name + "' WHERE ID = 'name'")
        self.configuration.project.execute("UPDATE tbl_info SET parameter = '" + self.opt_subject.currentText() + "' WHERE ID = 'subject'")
        self.configuration.project.execute("UPDATE tbl_info SET parameter = '" + self.opt_organ.currentText() + "' WHERE ID = 'organ'")
        self.configuration.project.execute("UPDATE tbl_info SET parameter = '" + self.opt_quantitative.currentText() + "' WHERE ID = 'quantitative'")

        self.accept()
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
