import sys
import os
from PyQt5 import QtWidgets

from marissa.toolbox.tools import tool_pydicom, tool_plot
from marissa.toolbox.creators import creator_dialog

class GUI(creator_dialog.Inheritance):
    def __init__(self, parent=None, config=None, SOPinstanceUID=None, segmentationID = None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        if SOPinstanceUID is None and not segmentationID is None:
            ID = self.configuration.project.select("SELECT SOPinstanceUId FROM tbl_segmentation WHERE segmentationID = " + str(segmentationID))[0][0]
        else:
            ID = SOPinstanceUID

        if not ID is None:
            dcm = self.configuration.project.get_data(ID)[0]
            pd = tool_pydicom.get_dcm_pixel_data(dcm, representation=True)
            self.mpl_plot.canvas.ax.imshow(pd, cmap="gray")

            if not segmentationID is None:
                selection = self.configuration.project.select("SELECT mask, points FROM tbl_segmentation WHERE segmentationID = " + str(segmentationID))[0]
                rgba_mask = tool_plot.masks2delta2rgba(selection[0], selection[0])
                self.mpl_plot.canvas.ax.imshow(rgba_mask[1])

                for parr in selection[1]:
                    y = parr[:,0]
                    x = parr[:,1]
                    self.mpl_plot.canvas.ax.plot(x, y, c="red")


            self.mpl_plot.canvas.ax.axis("off")
            self.mpl_plot.canvas.draw()

        return

    def closeEvent(self, event):
        event.accept()
        return

if __name__ == "__main__":
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())
