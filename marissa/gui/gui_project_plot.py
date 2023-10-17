import sys
import os
import numpy as np
import warnings
from PyQt5 import QtWidgets, QtCore

from marissa.gui import gui_project
from marissa.toolbox.creators import creator_gui
from marissa.toolbox.tools import tool_pydicom
from matplotlib import pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore")

class GUI(creator_gui.Inheritance):
    def __init__(self, parent=None, config=None):
        super().__init__(parent, config, os.path.basename(__file__).replace(".py", ""))

        selection = self.configuration.project.select("SELECT description FROM tbl_standardization_setup ORDER BY description")
        standardized_setups = [selection[i][0] for i in range(len(selection))]
        self.opt_setup.addItems(standardized_setups)

        self.counter = 0
        self.standardization_selection = None
        self.opt_setup.currentTextChanged.connect(self.opt_setup_clicked)
        self.btn_first.clicked.connect(self.btn_first_clicked)
        self.btn_previous.clicked.connect(self.btn_previous_clicked)
        self.btn_next.clicked.connect(self.btn_next_clicked)
        self.btn_last.clicked.connect(self.btn_last_clicked)
        self.btn_plot.clicked.connect(self.btn_plot_clicked)

        self.opt_setup_clicked(None)
        return

    # Override closeEvent
    def closeEvent(self, event):
        global gui_run
        gui_run = gui_project.GUI(None, self.configuration)
        gui_run.show()
        event.accept()
        return

    def opt_setup_clicked(self, value):
        selection = self._get_setup_info()
        setupID = selection[0]
        mode = selection[-1]
        if mode == "ensemble":
            self.standardization_selection = self.configuration.project.select("SELECT TSP.description, TSP.ordering, TSP.parameterID, TSP.VR, TSP.VM, TS.bin, TS.VMindex, TS.reference, TS.regressor, TS.rmse, TS.p, TS.rsquared, TS.x FROM tbl_standardization_parameter AS TSP INNER JOIN tbl_standardization AS TS ON (TS.setupID = TSP.setupID AND TS.parameterID = TSP.parameterID) WHERE TS.setupID = " + str(setupID) + " ORDER BY TSP.ordering, TS.VMindex, TS.bin, TS.x")
        else:
            self.standardization_selection = self.configuration.project.select("SELECT TSP.description, TSP.ordering, TSP.parameterID, TSP.VR, TSP.VM, TS.bin, TS.VMindex, TS.reference, TS.regressor, TS.rmse, TS.p, TS.rsquared, CAST(TS.x AS TEXT) FROM tbl_standardization_parameter AS TSP INNER JOIN tbl_standardization AS TS ON (TS.setupID = TSP.setupID AND TS.parameterID = TSP.parameterID) WHERE TS.setupID = " + str(setupID) + " ORDER BY TSP.ordering, TS.VMindex, TS.bin, TS.x")

        self.counter = 0
        self.plot(self.counter, None)
        return

    def btn_first_clicked(self):
        self.counter = 0
        self.plot(self.counter, None)
        return

    def btn_previous_clicked(self):
        if self.counter > 0:
            self.counter = self.counter - 1
            self.plot(self.counter, None)
        return

    def btn_next_clicked(self):
        if self.counter < self._get_count_plots()-1:
            self.counter = self.counter + 1
            self.plot(self.counter, None)
        return

    def btn_last_clicked(self):
        self.counter = self._get_count_plots()-1
        self.plot(self.counter, None)
        return

    def btn_plot_clicked(self):
        path_out = self.get_directory(None)
        if not path_out is None and not path_out=="":
            path_out = os.path.join(path_out, "MARISSA_training_export_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            for i in range(self._get_count_plots()):
                self.plot(i, path_out)
            self.show_dialog("The plots were successfully exported to " + path_out)
        return

    def _get_setup_info(self):
        setupID, bins, clustertype, regressiontype, ytype, mode = self.configuration.project.select("SELECT setupID, bins, clustertype, regressiontype, ytype, mode FROM tbl_standardization_setup WHERE description = '" + self.opt_setup.currentText() + "'")[0]
        return setupID, bins, clustertype, regressiontype, ytype, mode

    def _get_count_plots(self):
        selection = self._get_setup_info()
        setupID = selection[0]
        mode = selection[-1]
        if mode == "ensemble":
            counter = selection[1] # number of bins
        else:
            counter = self.configuration.project.select("SELECT COUNT(setupID) FROM tbl_standardization WHERE setupID = " + str(setupID))[0][0]
        return counter

    def plot(self, plot_num, path_save=None):
        # get ax as pyqt canvas plot or save as figure
        if path_save is None:
            self.lbl_num.setText(str(plot_num+1) + " of " + str(self._get_count_plots()))
            ax = self.mpl_plot.new_plot()
        else:
            fig = plt.figure(figsize=(16,8))
            ax = fig.add_subplot(111)

        # create plot
        setupID, bins, clustertype, regressiontype, ytype, mode = self._get_setup_info()
        parameter_description, ordering, parameterID, VR, VM, bin, VMindex, reference, regressor, rmse, p, rsquared, x = self.standardization_selection[plot_num]
        if mode == "ensemble":
            bin = plot_num + 1

        exec("from marissa.modules.regression import " + regressiontype + " as regression")
        exec("from marissa.modules.clustering import " + clustertype + " as clustering")
        cm = eval("clustering.Model(bins=bins)")
        rm = eval("regression.Model(ytype=ytype, load=regressor)")

        selection_data = self.configuration.project.select("SELECT TSD.segmentedvalues, TSD.parameters FROM tbl_standardization_data AS TSD INNER JOIN tbl_standardization_match_data_setup_parameter AS TSMDSP ON (TSD.setupID = TSMDSP.setupID AND TSD.segmentationID = TSMDSP.segmentationID AND TSD.SOPinstanceUID = TSMDSP.SOPinstanceUID) WHERE TSD.setupID=" + str(setupID) + " AND VMindex=" + str(VMindex) + " AND parameterID=" + str(parameterID))
        segmentedvalues = []
        parametervalues = []
        binindex = []
        for j in range(len(selection_data)):
            if bins > 1:
                cmrun = cm.run(selection_data[j][0])
                for jj in range(len(cmrun)):
                    try:
                        segmentedvalues.append(cmrun[jj])
                        binindex.append([jj+1] * len(cmrun[jj]))
                    except:
                        pass # if empty
            else:
                segmentedvalues.append([selection_data[j][0]])

            if mode == "ensemble":
                parametervalues.append([selection_data[j][1]] * len(selection_data[j][0]))
            elif VM == 1:
                parametervalues.append([selection_data[j][1][ordering-1]] * len(selection_data[j][0]))
            else:
                parametervalues.append([selection_data[j][1][ordering-1][VMindex-1]] * len(selection_data[j][0]))

        segmentedvalues = np.concatenate(np.array(segmentedvalues).flatten())
        parametervalues = np.concatenate(np.array(parametervalues).flatten())

        if bins > 1:
            indeces = np.argwhere(np.concatenate(np.array(binindex).flatten()) == bin).flatten()
            segmentedvalues = segmentedvalues[indeces].flatten()
            try: # ensemble mode
                parametervalues = parametervalues[indeces, :]
            except:
                parametervalues = parametervalues[indeces].flatten()

        if mode == "ensemble":
            selection_grouping = self.configuration.project.select("SELECT TS.x FROM tbl_standardization AS TS INNER JOIN tbl_standardization_parameter AS TSP ON (TSP.setupID = TS.setupID AND TSP.parameterID = TS.parameterID) WHERE TS.setupID = " + str(setupID) + " AND TS.bin = " + str(bin) + " ORDER BY TSP.ordering, TS.VMindex")

            reference_setting = []
            for i in range(len(selection_grouping)):
                if not selection_grouping[i][0] is None:
                    parametervalues[:, i] = np.array([selection_grouping[i][0].index(val) for val in parametervalues[:, i]])
                    reference_setting.append(0)
                else:
                    ref = self.configuration.project.select("SELECT TS.reference FROM tbl_standardization AS TS INNER JOIN tbl_standardization_parameter AS TSP ON (TSP.setupID = TS.setupID AND TSP.parameterID = TS.parameterID) WHERE TS.setupID = " + str(setupID) + " ORDER BY TSP.ordering, TS.VMindex LIMIT 1 OFFSET " + str(i))[0][0]
                    reference_setting.append(float(ref))

            parametervalues = np.array(parametervalues).astype(float)
            standardizedvalues = rm.apply(parametervalues, segmentedvalues, ytype)

            reference_indeces = np.where(np.all(parametervalues==np.array(reference_setting), axis=1))[0]
            reference_y = np.mean(segmentedvalues[reference_indeces])

            if ytype == "absolute":
                dyo = segmentedvalues - reference_y
                dys = standardizedvalues - reference_y
            else: # relative
                dyo = 100 * (segmentedvalues - reference_y) / reference_y
                dys = 100 * (standardizedvalues - reference_y) / reference_y

            import seaborn
            import pandas

            df = pandas.DataFrame(np.transpose(np.array([np.array([0] * len(dyo)), dyo]))[:1000,:], columns=["x", "y"])

            #seaborn.violinplot(df["y"], ax=ax)
            seaborn.swarmplot(df, y="y", ax=ax, color="blue")

            ax.scatter([0] , [0], c="green", label="reference", zorder=10, marker="x")
            ax.axhline(0, c="green", ls="--")
            ax.axvline(0, c="green", ls="--")

            if bins > 1:
                ax.set_title("All confounding parameters at bin " + str(bin) + " on " + str(len(selection_data)) + " datasets with " + str(len(segmentedvalues)) + " datapoints\n" + regressiontype + " regression on " + ytype + " error with " + clustertype + " clustering for " + str(bins) + " bins in " + mode + " mode")
            else:
                ax.set_title("All confounding parameters on " + str(len(selection_data)) + " datasets with " + str(len(segmentedvalues)) + " datapoints\n" + regressiontype + " regression on " + ytype + " error with " + clustertype + " clustering for " + str(bins) + " bins in " + mode + " mode")

        else:
            if tool_pydicom.get_VR_type(VR) == str:
                index_reference = np.argwhere(np.array(parametervalues).astype(str) == reference).flatten()
                index_confounder = np.argwhere(np.array(parametervalues).astype(str) == x).flatten()

                reference_y = np.mean(segmentedvalues[index_reference].flatten())
                confounder_y = np.mean(segmentedvalues[index_confounder].flatten())

                if reference_y == 0:
                    reference_y = 1e-6

                if ytype == "absolute":
                    dyc = segmentedvalues[index_confounder] - reference_y
                    dyr = segmentedvalues[index_reference] - reference_y
                else: # relative
                    dyc = 100 * (segmentedvalues[index_confounder] - reference_y) / reference_y
                    dyr = 100 * (segmentedvalues[index_reference] - reference_y) / reference_y

                dx_regression = np.array([0, 1])
                dy_regression = rm.predict(dx_regression.reshape([-1,1]))

                ax.scatter([0] * len(dyr), dyr, s=2, c="blue")
                ax.scatter([1] * len(dyc), dyc, s=2, c="blue")
                ax.plot(dx_regression, dy_regression, c="red", label="regression")
                ax.scatter(dx_regression, dy_regression, c="red")
                ax.axhline(0, c="green", ls="--")
                ax.axvline(0, c="green", ls="--")
                ax.scatter(0, 0, c="green", label="reference", marker="x")
                ax.set_xticks([0, 1], labels=[reference + "\n(reference)", x])
            else:
                index_reference = np.argwhere(np.array(parametervalues).astype(float) == float(reference)).flatten()
                reference_y = np.mean(segmentedvalues[index_reference].flatten())
                if reference_y == 0:
                    reference_y = 1e-6

                if ytype == "absolute":
                    dy = segmentedvalues - reference_y
                else: # relative
                    dy = 100 * (segmentedvalues - reference_y) / reference_y

                dx_regression = np.linspace(np.min(parametervalues), np.max(parametervalues), num=10000)
                dy_regression = rm.predict(dx_regression.reshape([-1,1]))
                dx_regression_unique = np.unique(parametervalues)
                dy_regression_unique = rm.predict(dx_regression_unique.reshape((-1,1)))

                ax.scatter(parametervalues, dy, s=2, c="blue")
                ax.scatter(dx_regression_unique, dy_regression_unique, c="red")
                ax.plot(dx_regression, dy_regression, c="red", label="regression")
                ax.axhline(0, c="green", ls="--")
                ax.axvline(float(reference), c="green", ls="--")
                ax.scatter(float(reference), 0, c="green", label="reference", marker="x")

            # general
            if bins > 1:
                ax.set_title(parameter_description + " at bin " + str(bin) + " for VMindex " + str(VMindex) + " on " + str(len(selection_data)) + " datasets with " + str(len(segmentedvalues)) + " datapoints\n" + regressiontype + " regression on " + ytype + " error with " + clustertype + " clustering for " + str(bins) + " bins in " + mode + " mode")
            else:
                ax.set_title(parameter_description + " for VMindex " + str(VMindex) + " on " + str(len(selection_data)) + " datasets with " + str(len(segmentedvalues)) + " datapoints\n" + regressiontype + " regression on " + ytype + " error in " + mode + " mode")

            ax.set_xlabel(parameter_description)

        ax.set_ylabel(ytype + " error")

        ax.axis("on")
        ax.legend(loc="upper center")

        # show or save the plot
        if path_save is None:
            ax.autoscale(enable=True)
            self.mpl_plot.draw()
        else:
            plt.tight_layout()
            path_out = os.path.join(path_save, parameter_description, "VMindex" + str(VMindex) + "Bin" + str(bin))
            os.makedirs(path_out, exist_ok=True)
            if tool_pydicom.get_VR_type(VR) == str:
                file_path = "regression_" + x + ".jpg"
            else:
                file_path = "regression.jpg"
            plt.savefig(os.path.join(path_out, file_path), dpi=300)
        return

if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    global gui_run
    app = QtWidgets.QApplication(sys.argv)
    gui_run = GUI()
    gui_run.show()
    sys.exit(app.exec_())