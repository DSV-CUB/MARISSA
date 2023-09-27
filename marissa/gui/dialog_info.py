import sys
import os
import webbrowser
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_dialog

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        str_Info = "Version:\t\t" + self.configuration.database.select("SELECT parameter FROM tbl_info WHERE ID = 'version'")[0][0] + "\n"
        str_Info = str_Info + "Date:\t\t" + self.configuration.database.select("SELECT parameter FROM tbl_info WHERE ID = 'timestamp'")[0][0] + "\n"
        #str_Info = str_Info + "\n"
        str_Info = str_Info + "Author:\t\t" + self.configuration.database.select("SELECT parameter FROM tbl_info WHERE ID = 'author'")[0][0] + "\n"
        str_Info = str_Info + "Contact:\t\t" + self.configuration.database.select("SELECT parameter FROM tbl_info WHERE ID = 'contact'")[0][0].replace("\n", "\n\t\t")

        self.lbl_Info.setText(str_Info)
        self.lbl_Info.mousePressEvent = self.lbl_Info_clicked
        return

    def closeEvent(self, event):
        event.accept()
        return

    def lbl_Info_clicked(self, event):
        start = self.lbl_Info.text().find("http")
        if start >= 0:
            end = self.lbl_Info.text().find("\n", start)
            if end > 0:
                url = self.lbl_Info.text()[start:end+1]
            else:
                url = self.lbl_Info.text()[start:]
            webbrowser.open_new_tab(url)
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
