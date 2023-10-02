import sys
import os
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_gui
from marissa.gui import dialog_info, gui_start, gui_project_data, gui_project_setup, gui_project_train, gui_project_standardize, gui_project_famd, gui_project_plot

class GUI(creator_gui.Inheritance):

    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        self.setWindowTitle("MARISSA - Project : " + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])

        self.btn_icon_database_light.clicked.connect(self.btn_icon_database_light_clicked)
        self.btn_icon_famd_light.clicked.connect(self.btn_icon_famd_light_clicked)
        self.btn_icon_flow_light.clicked.connect(self.btn_icon_flow_light_clicked)
        self.btn_icon_logo_info.clicked.connect(self.btn_icon_logo_info_clicked)

        self.btn_setup.clicked.connect(self.btn_setup_clicked)
        self.btn_data.clicked.connect(self.btn_data_clicked)
        self.btn_train.clicked.connect(self.btn_train_clicked)
        self.btn_standardize.clicked.connect(self.btn_standardize_clicked)
        self.btn_change.clicked.connect(self.btn_change_clicked)
        return

    def closeEvent(self, event):
        event.accept()
        return

    def btn_change_clicked(self):
        self.configuration.project = None
        global gui_run
        gui_run = gui_start.GUI(None, self.configuration)
        gui_run.show()
        self.close()
        return

    def btn_setup_clicked(self):
        global gui_run
        gui_run = gui_project_setup.GUI(None, self.configuration)
        gui_run.show()
        self.close()
        return

    def btn_data_clicked(self):
        global gui_run
        gui_run = gui_project_data.GUI(None, self.configuration)
        gui_run.show()
        self.close()
        return

    def btn_train_clicked(self):
        if self.configuration.project.setup_exist():
            global gui_run
            gui_run = gui_project_train.GUI(None, self.configuration)
            gui_run.show()
            self.close()
        else:
            self.show_dialog("There is no setup in this project. Please create a setup first.", "Warning")
        return

    def btn_standardize_clicked(self):
        if self.configuration.project.standardization_exist():
            global gui_run
            gui_run = gui_project_standardize.GUI(None, self.configuration)
            gui_run.show()
            self.close()
        else:
            self.show_dialog("There is no trained standardization in this project. Please train for a setup first.", "Warning")

        return

    def btn_icon_famd_light_clicked(self):
        if self.configuration.project.setup_exist():
            try:
                global gui_run
                gui_run = gui_project_famd.GUI(None, self.configuration)
                gui_run.show()
                self.close()
            except:
                self.show_dialog("FAMD analysis module could not be started, probably you need to install R and necessary packages (NbClust, FactoMineR, factoextra)", "Warning")
        else:
            self.show_dialog("There is no setup in this project. Please create a setup first.", "Warning")
        return

    def btn_icon_flow_light_clicked(self):
        if self.configuration.project.standardization_exist():
            global gui_run
            gui_run = gui_project_plot.GUI(None, self.configuration)
            gui_run.show()
            self.close()
        else:
            self.show_dialog("There is no trained standardization in this project. Please train for a setup first.", "Warning")
        return

    def btn_icon_database_light_clicked(self):
        path_to = self.get_directory(None)
        if not path_to is None and not path_to == "":
            worked = self.configuration.project.plot_database_structure(path_to, "dbstructure_" + self.configuration.project.select("SELECT parameter FROM tbl_info WHERE ID = 'name'")[0][0])
            if not worked:
                self.show_dialog("The Database structure could not be plotted. Probably there is something wrong with pygraphviz or sqlalchemy. In order to get more information, please use an IDE with debugging.", "Critical")
            else:
                self.show_dialog("The Database structure was successfully plotted.", "Information")
        return

    def btn_icon_logo_info_clicked(self):
        dialog = dialog_info.GUI(self, self.configuration)
        dialog.exec()
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
