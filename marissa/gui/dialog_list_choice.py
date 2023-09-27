import sys
import os
from PyQt5 import QtWidgets

from marissa.toolbox.creators import creator_dialog

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None, **kwargs):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        description = kwargs.get("description", "")
        listvalues = kwargs.get("list", [])
        listchoice = kwargs.get("listchoice", "single")

        self.lbl_description.setText(description)

        if listchoice == "single":
            self.lst_choice.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        else:
            self.lst_choice.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        self.lst_choice.addItems(listvalues)

        self.btn_select.clicked.connect(self.btn_select_clicked)

        self.result = None
        return

    def closeEvent(self, event):
        event.accept()
        return

    def btn_select_clicked(self):
        selecteditems = self.lst_choice.selectedItems()

        if len(selecteditems) > 0:
            self.result = []

            for item in selecteditems:
                self.result.append(item.text())

            if len(self.result) == 0:
                self.result = None
            elif self.lst_choice.selectionMode == QtWidgets.QAbstractItemView.SingleSelection and len(self.result) > 0:
                self.result = self.result[0]

            self.accept()
        return


if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
