from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy import stats

import warnings
warnings.filterwarnings('ignore')




# SETUP
project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\DoktorarbeitDSV.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER"

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]
reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]

colours = ["#008000", "#ff8000", "#ff0000"]
cohorts = ["Healthy", "HCM", "Amyloidosis"]



def plot_roc(ax, roc_analysis, c="blue", optimum_colour="#ff0000", optimum_text=True, label="", zorder=10, thick=False):
    if c=="C3":
        c="C9"
    if optimum_colour == "C3":
        optimum_colour="C9"

    ax.plot(roc_analysis["roc_fpr"] * 100, roc_analysis["roc_tpr"] * 100, c=c, label=label, zorder=zorder, lw=(4 if thick else 2))
    if not optimum_colour is None:
        ax.scatter([roc_analysis["optimal_roc_point"][0] * 100], [roc_analysis["optimal_roc_point"][1] * 100], edgecolor=optimum_colour, facecolor="None", lw=(4 if thick else 2), s=(100 if thick else 80), zorder=zorder)
        if optimum_text:
            ax.text(roc_analysis["optimal_roc_point"][0] * 100 + 1, roc_analysis["optimal_roc_point"][1] * 100 - 1, "optimum\nthreshold = " + "{:.2f}".format(np.round(roc_analysis["optimal_threshhold"], 2)) + " ms\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + " %\nspecificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", c=optimum_colour, horizontalalignment="left", verticalalignment="top", zorder=zorder, fontsize=14)
    return

def plot_roc_overhead(ax):
    ax.plot([0, 100], [0, 100], c="#66666666")
    ax.set_ylabel("sensitivity [%]", fontsize=18)
    ax.set_xlabel("1 - specificity [%]", fontsize=18)
    ax.set_ylim((-2, 102))
    ax.set_xlim((-2, 102))
    return


# PREPARATION
project = marissadb.Module(project_path)

pids = []
for pd in parameters:
    pids.append(project.select("SELECT parameterID FROM tbl_parameter WHERE description = '" + pd + "'")[0][0])

selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TRAINING'")
data = []
info_data = []
for i in range(len(selection)):
    data.append([selection[i][0], selection[i][1]])
    info_data.append(project.get_data_parameters(selection[i][0], pids))
info_data = np.array(info_data)

selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TESTHEALTHY'")
data_test = []
info_data_test = []
for i in range(len(selection)):
    data_test.append([selection[i][0], selection[i][1]])
    info_data_test.append(project.get_data_parameters(selection[i][0], pids))
info_data_test = np.array(info_data_test)

# HCM Patient
selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TESTPATIENTHCM'")
data_patient_hcm = []
info_data_patient_hcm = []

for i in range(len(selection)):
    dcm = project.get_data(selection[i][0])[0]
    data_patient_hcm.append([selection[i][0], selection[i][1]])
    info_data_patient_hcm.append(project.get_data_parameters(selection[i][0], pids))
info_data_patient_hcm = np.array(info_data_patient_hcm)

# Amyloidose Patient
selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TESTPATIENTAMYLOIDOSE'")
data_patient_amy = []
info_data_patient_amy = []
for i in range(len(selection)):
    data_patient_amy.append([selection[i][0], selection[i][1]])
    info_data_patient_amy.append(project.get_data_parameters(selection[i][0], pids))
info_data_patient_amy = np.array(info_data_patient_amy)

# load training
file_s2 = os.path.join(path_out, "train_step2_top3_binning.txt")
file = open(file_s2, "r")
readdata = file.readlines()
file.close()
setup_info = [readdata[ii].split("\t") for ii in range(1, len(readdata))]
evalS2 = np.array(setup_info)[:,-1].flatten().astype(float)
index_top = np.argsort(evalS2)[0] # lowest CoV is best
best_setup = setup_info[index_top]

setupID = project.select("SELECT setupID FROM tbl_setup WHERE description = '" + best_setup[0] + "'")[0][0]


########################################################################################################################
# PROGRESSION & BOX PLOT ###############################################################################################
########################################################################################################################
all_y_healthy = []
all_y_healthy_train = []
all_y_patient_hcm = []
all_y_patient_amy = []

# PLOT 1: Healthy Volunteers
for i in range(len(data_test)):
    dcm = project.get_data(data_test[i][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data_test[i][1]))[0][0]
    try:
        md = project.get_standardization(dcm, mask, setupID, False)
    except:
        continue

    prog_y = []
    for j in range(len(md.value_progression)):
        y = np.mean(md.value_progression[j])
        prog_y.append(y)
    all_y_healthy.append(prog_y)

# PLOT 2: PATIENT HCM
for i in range(len(data_patient_hcm)):
    dcm = project.get_data(data_patient_hcm[i][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data_patient_hcm[i][1]))[0][0]
    try:
        md = project.get_standardization(dcm, mask, setupID, False)
    except:
        continue

    prog_y = []
    for j in range(len(md.value_progression)):
        y = np.mean(md.value_progression[j])
        prog_y.append(y)

    all_y_patient_hcm.append(prog_y)

# PLOT 3: PATIENT AMYLOIDOSE
for i in range(len(data_patient_amy)):
    dcm = project.get_data(data_patient_amy[i][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data_patient_amy[i][1]))[0][0]
    try:
        md = project.get_standardization(dcm, mask, setupID, False)
    except:
        continue

    prog_y = []
    for j in range(len(md.value_progression)):
        y = np.mean(md.value_progression[j])
        prog_y.append(y)
    all_y_patient_amy.append(prog_y)

# healthy train data
for i in range(len(data)):
    dcm = project.get_data(data[i][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data[i][1]))[0][0]
    try:
        md = project.get_standardization(dcm, mask, setupID, False)
    except:
        continue

    prog_y = []
    for j in range(len(md.value_progression)):
        y = np.mean(md.value_progression[j])
        prog_y.append(y)
    all_y_healthy_train.append(prog_y)


healthy_before = np.array(all_y_healthy)[:,0].flatten()
healthy_after = np.array(all_y_healthy)[:,-1].flatten()
healthy_train_before = np.array(all_y_healthy_train)[:,0].flatten()
healthy_train_after = np.array(all_y_healthy_train)[:,-1].flatten()
patient_before_hcm = np.array(all_y_patient_hcm)[:,0].flatten()
patient_after_hcm = np.array(all_y_patient_hcm)[:,-1].flatten()
patient_before_amy = np.array(all_y_patient_amy)[:,0].flatten()
patient_after_amy = np.array(all_y_patient_amy)[:,-1].flatten()

#healthy_all_before = np.hstack((healthy_train_before, healthy_before))
#healthy_all_after = np.hstack((healthy_train_after, healthy_after))

healthy_all_before = healthy_before
healthy_all_after = healthy_after



cmap = cm.get_cmap('nipy_spectral')

fig = plt.figure(None, figsize=(16, 24), constrained_layout=True)
subfigs = fig.subfigures(3, 1)#, width_ratios = [0.05, 1, 1, 0.05], height_ratios = [1, 1, 1])
axes_sf0 = subfigs[0].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1]})
axes_sf1 = subfigs[1].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1]})
axes_sf2 = subfigs[2].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1]})
ax = np.vstack((axes_sf0, axes_sf1, axes_sf2))

subfigs[0].supylabel("Healthy vs. HCM", fontsize=20, fontweight="bold")
subfigs[1].supylabel("Healthy vs. AMY", fontsize=20, fontweight="bold")
subfigs[2].supylabel("HCM vs. AMY", fontsize=20, fontweight="bold")


ax[0,0].set_title("before standardization", fontsize=20, fontweight="bold")
ax[0,1].set_title("after standardization", fontsize=20, fontweight="bold")

for ii in range(3):
    for iii in range(2):
        plot_roc_overhead(ax[ii,iii])


# BEFORE STANDARDIZATION
parameter_healthy = [" | ".join(info_data[:, 2:].tolist()[ii]) for ii in range(len(info_data))] + [" | ".join(info_data_test[:, 2:].tolist()[ii]) for ii in range(len(info_data_test))]

parameter_HCM = [" | ".join(info_data_patient_hcm[:, 2:].tolist()[ii]) for ii in range(len(info_data_patient_hcm))]
parameter_AMY = [" | ".join(info_data_patient_amy[:, 2:].tolist()[ii]) for ii in range(len(info_data_patient_amy))]
unique_settings = np.unique(parameter_HCM + parameter_AMY)
all_unique_settings = np.unique(parameter_healthy + parameter_HCM + parameter_AMY)

parameter_healthy = [" | ".join(info_data_test[:, 2:].tolist()[ii]) for ii in range(len(info_data_test))]


for uhcm in np.unique(parameter_HCM):
    indeces_hcm = np.argwhere(np.array(parameter_HCM)==uhcm).flatten()
    indeces = np.argwhere(np.array(parameter_healthy)==uhcm).flatten()
    ic = np.argwhere(all_unique_settings==uhcm).flatten()[0]
    color = cmap(ic/len(all_unique_settings))
    #color = "C" + str(np.argwhere(unique_settings==uhcm).flatten()[0] + 1)

    if len(indeces) > 0:
        roc_analysis = tool_statistics.roc_curve_analysis(patient_before_hcm[indeces_hcm], healthy_all_before[indeces])
        plot_roc(ax[0,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        roc_analysis = tool_statistics.roc_curve_analysis(patient_after_hcm[indeces_hcm], healthy_all_after[indeces])
        plot_roc(ax[0,1], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

    indeces = np.argwhere(np.array(parameter_AMY)==uhcm).flatten()
    if len(indeces) > 0:
        roc_analysis = tool_statistics.roc_curve_analysis(patient_before_amy[indeces], patient_before_hcm[indeces_hcm])
        plot_roc(ax[2,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        roc_analysis = tool_statistics.roc_curve_analysis(patient_after_amy[indeces], patient_after_hcm[indeces_hcm])
        plot_roc(ax[2,1], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)


for uamy in np.unique(parameter_AMY):
    indeces_amy = np.argwhere(np.array(parameter_AMY)==uamy).flatten()
    indeces = np.argwhere(np.array(parameter_healthy)==uamy).flatten()

    ic = np.argwhere(all_unique_settings==uamy).flatten()[0]
    color = cmap(ic/len(all_unique_settings))
    #color = "C" + str(np.argwhere(unique_settings==uamy).flatten()[0] + 1)

    if len(indeces) > 0:
        roc_analysis = tool_statistics.roc_curve_analysis(patient_before_amy[indeces_amy], healthy_all_before[indeces])
        plot_roc(ax[1,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uamy + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        roc_analysis = tool_statistics.roc_curve_analysis(patient_after_amy[indeces_amy], healthy_all_after[indeces])
        plot_roc(ax[1,1], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uamy + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)




# AFTER STANDARDIZATION
roc_analysis_healthy_hcm = tool_statistics.roc_curve_analysis(patient_after_hcm, healthy_after)
roc_analysis_healthy_amy = tool_statistics.roc_curve_analysis(patient_after_amy, healthy_after)
roc_analysis_hcm_amy = tool_statistics.roc_curve_analysis(patient_after_amy, patient_after_hcm)

plot_roc(ax[0,1], roc_analysis_healthy_hcm, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_healthy_hcm["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_healthy_hcm["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)
plot_roc(ax[1,1], roc_analysis_healthy_amy, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_healthy_amy["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_healthy_amy["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)
plot_roc(ax[2,1], roc_analysis_hcm_amy, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_hcm_amy["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_hcm_amy["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)

legend = ax[0,0].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)
legend = ax[1,0].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)
legend = ax[2,0].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)

legend = ax[0,1].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)
legend = ax[1,1].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)
legend = ax[2,1].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)


for i in range(3):
    for j in range(2):
        ax[i,j].xaxis.set_tick_params(labelsize=18)
        ax[i,j].yaxis.set_tick_params(labelsize=18)


#plt.tight_layout()
plt.savefig(os.path.join(path_out, "roc_analysis.jpg"), dpi=300)