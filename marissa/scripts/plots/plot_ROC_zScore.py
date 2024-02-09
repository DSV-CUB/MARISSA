from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
from marissa.scripts.plots import basic_functions
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

########################################################################################################################
# USER INPUT ###########################################################################################################
########################################################################################################################

project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\DoktorarbeitDSV.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER"
path_training = os.path.join(path_out, "train_step2_top3_binning.txt")

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]
reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]

colours = ["#008000", "#ff8000", "#ff0000"]
cohorts = ["Healthy", "HCM", "Amyloidosis"]

########################################################################################################################
# PREPARATION ##########################################################################################################
########################################################################################################################

project = marissadb.Module(project_path)
setupID = basic_functions.get_BPSP_setupID(project, path_training)
pids = basic_functions.get_pids(project, parameters)

data, info_data = basic_functions.get_data_info(project, "TRAINING", pids)
data_test, info_data_test = basic_functions.get_data_info(project, "TESTHEALTHY", pids)
data_patient_hcm, info_data_patient_hcm = basic_functions.get_data_info(project, "TESTPATIENTHCM", pids)
data_patient_amy, info_data_patient_amy = basic_functions.get_data_info(project, "TESTPATIENTAMYLOIDOSE", pids)

all_y_healthy = basic_functions.get_all_y(project, data_test, setupID)
all_y_healthy_train = basic_functions.get_all_y(project, data, setupID)
all_y_patient_hcm = basic_functions.get_all_y(project, data_patient_hcm, setupID)
all_y_patient_amy = basic_functions.get_all_y(project, data_patient_amy, setupID)

healthy_all_before = np.array(all_y_healthy)[:,0].flatten()
healthy_all_after = np.array(all_y_healthy)[:,-1].flatten()
healthy_train_before = np.array(all_y_healthy_train)[:,0].flatten()
healthy_train_after = np.array(all_y_healthy_train)[:,-1].flatten()
patient_before_hcm = np.array(all_y_patient_hcm)[:,0].flatten()
patient_after_hcm = np.array(all_y_patient_hcm)[:,-1].flatten()
patient_before_amy = np.array(all_y_patient_amy)[:,0].flatten()
patient_after_amy = np.array(all_y_patient_amy)[:,-1].flatten()

# Z SCORE
train_settings = info_data[:, 2:]
train_unique_settings = np.unique(train_settings, axis=0)
z_mean = []
z_std = []
for i in range(len(train_unique_settings)):
    indeces = np.argwhere(np.all(train_settings==train_unique_settings[i,:], axis=1)).flatten()
    print(str(train_unique_settings[i,:]) + " : " +  str(len(indeces)))
    z_mean.append(np.mean(np.array(all_y_healthy_train)[:,0].flatten()[indeces]))
    z_std.append(np.std(np.array(all_y_healthy_train)[:,0].flatten()[indeces]))
z_mean = np.array(z_mean)
z_std = np.array(z_std)

def get_z_score(project, data, info, setupID, zmean, zstd, settings):
    z_score = []
    for i in range(len(data)):
        dcm = project.get_data(data[i][0])[0]
        mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data[i][1]))[0][0]
        try:
            md = project.get_standardization(dcm, mask, setupID, False)
        except:
            continue

        # zscore
        try:
            index = np.argwhere(np.all(settings==info[i, 2:], axis=1)).flatten()[0]
            z_score.append(np.mean((md.value_progression[0] - z_mean[index]) / z_std[index]))
        except:
            z_score.append(np.nan)
    return np.array(z_score)

z_score_healthy = get_z_score(project, data_test, info_data_test, setupID, z_mean, z_std, train_unique_settings)
z_score_hcm = get_z_score(project, data_patient_hcm, info_data_patient_hcm, setupID, z_mean, z_std, train_unique_settings)
z_score_amy = get_z_score(project, data_patient_amy, info_data_patient_amy, setupID, z_mean, z_std, train_unique_settings)

########################################################################################################################
# PROGRESSION & BOX PLOT ###############################################################################################
########################################################################################################################

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


cmap = cm.get_cmap('nipy_spectral')

fig = plt.figure(None, figsize=(8, 24), constrained_layout=True)
subfigs = fig.subfigures(3, 1)#, width_ratios = [0.05, 1, 1, 0.05], height_ratios = [1, 1, 1])
axes_sf0 = subfigs[0].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1]})
axes_sf1 = subfigs[1].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1]})
axes_sf2 = subfigs[2].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1]})
ax = np.vstack((axes_sf0, axes_sf1, axes_sf2))

subfigs[0].supylabel("HTE vs. HCM", fontsize=20, fontweight="bold")
subfigs[1].supylabel("HTE vs. AMY", fontsize=20, fontweight="bold")
subfigs[2].supylabel("HCM vs. AMY", fontsize=20, fontweight="bold")


ax[0,0].set_title("with z-Score", fontsize=20, fontweight="bold")

for ii in range(3):
    for iii in range(1):
        plot_roc_overhead(ax[ii,iii])

# INTRA-SCANNER-INTRA-SEQUENCE PLOTS
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

    # Healthy s HCM
    if len(indeces) > 0:
        #roc_analysis = tool_statistics.roc_curve_analysis(patient_before_hcm[indeces_hcm], healthy_all_before[indeces])
        #plot_roc(ax[0,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        #roc_analysis = tool_statistics.roc_curve_analysis(patient_after_hcm[indeces_hcm], healthy_all_after[indeces])
        #plot_roc(ax[0,1], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        roc_analysis = tool_statistics.roc_curve_analysis(z_score_hcm[indeces_hcm], z_score_healthy[indeces])
        plot_roc(ax[0,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

    #HCM vs AMY
    indeces = np.argwhere(np.array(parameter_AMY)==uhcm).flatten()
    if len(indeces) > 0:
        #roc_analysis = tool_statistics.roc_curve_analysis(patient_before_amy[indeces], patient_before_hcm[indeces_hcm])
        #plot_roc(ax[2,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        #roc_analysis = tool_statistics.roc_curve_analysis(patient_after_amy[indeces], patient_after_hcm[indeces_hcm])
        #plot_roc(ax[2,1], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        roc_analysis = tool_statistics.roc_curve_analysis(z_score_amy[indeces], z_score_hcm[indeces_hcm])
        plot_roc(ax[2,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uhcm + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)


for uamy in np.unique(parameter_AMY):
    indeces_amy = np.argwhere(np.array(parameter_AMY)==uamy).flatten()
    indeces = np.argwhere(np.array(parameter_healthy)==uamy).flatten()

    ic = np.argwhere(all_unique_settings==uamy).flatten()[0]
    color = cmap(ic/len(all_unique_settings))

    # Healthy vs AMY
    if len(indeces) > 0:
        #roc_analysis = tool_statistics.roc_curve_analysis(patient_before_amy[indeces_amy], healthy_all_before[indeces])
        #plot_roc(ax[1,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uamy + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        #roc_analysis = tool_statistics.roc_curve_analysis(patient_after_amy[indeces_amy], healthy_all_after[indeces])
        #plot_roc(ax[1,1], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uamy + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

        roc_analysis = tool_statistics.roc_curve_analysis(z_score_amy[indeces_amy], z_score_healthy[indeces])
        plot_roc(ax[1,0], roc_analysis, c=color, optimum_colour=color, optimum_text=False,  label=uamy + "\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10)

# AFTER STANDARDIZATION ONLY
#roc_analysis_healthy_hcm = tool_statistics.roc_curve_analysis(patient_after_hcm, healthy_all_after)
#roc_analysis_healthy_amy = tool_statistics.roc_curve_analysis(patient_after_amy, healthy_all_after)
#roc_analysis_hcm_amy = tool_statistics.roc_curve_analysis(patient_after_amy, patient_after_hcm)

#plot_roc(ax[0,1], roc_analysis_healthy_hcm, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_healthy_hcm["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_healthy_hcm["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)
#plot_roc(ax[1,1], roc_analysis_healthy_amy, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_healthy_amy["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_healthy_amy["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)
#plot_roc(ax[2,1], roc_analysis_hcm_amy, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_hcm_amy["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_hcm_amy["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)

# AFTER Z SCORE ONLY
z_score_amy = z_score_amy[np.logical_not(np.isnan(z_score_amy))]
roc_analysis_healthy_hcm = tool_statistics.roc_curve_analysis(z_score_hcm, z_score_healthy)
roc_analysis_healthy_amy = tool_statistics.roc_curve_analysis(z_score_amy, z_score_healthy)
roc_analysis_hcm_amy = tool_statistics.roc_curve_analysis(z_score_amy, z_score_hcm)

plot_roc(ax[0,0], roc_analysis_healthy_hcm, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_healthy_hcm["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_healthy_hcm["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)
plot_roc(ax[1,0], roc_analysis_healthy_amy, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_healthy_amy["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_healthy_amy["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)
plot_roc(ax[2,0], roc_analysis_hcm_amy, c="#008800", optimum_colour="#008800", optimum_text=False,  label="all scanners | all sequence variants\nsensitivity = " + "{:.2f}".format(np.round(roc_analysis_hcm_amy["optimal_roc_point"][1] * 100, 2)) + "% | specificity = " + "{:.2f}".format(np.round(100 - roc_analysis_hcm_amy["optimal_roc_point"][0] * 100, 2)) + " %", zorder=10, thick=True)

# ADD LEGENDS TO SUBPLOTS
legend = ax[0,0].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)
legend = ax[1,0].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)
legend = ax[2,0].legend(loc="lower center", fontsize=14)
legend.get_frame().set_facecolor('#efefef66')
legend.set_zorder(100)

#legend = ax[0,1].legend(loc="lower center", fontsize=14)
#legend.get_frame().set_facecolor('#efefef66')
#legend.set_zorder(100)
#legend = ax[1,1].legend(loc="lower center", fontsize=14)
#legend.get_frame().set_facecolor('#efefef66')
#legend.set_zorder(100)
#legend = ax[2,1].legend(loc="lower center", fontsize=14)
#legend.get_frame().set_facecolor('#efefef66')
#legend.set_zorder(100)

#legend = ax[0,2].legend(loc="lower center", fontsize=14)
#legend.get_frame().set_facecolor('#efefef66')
#legend.set_zorder(100)
#legend = ax[1,2].legend(loc="lower center", fontsize=14)
#legend.get_frame().set_facecolor('#efefef66')
#legend.set_zorder(100)
#legend = ax[2,2].legend(loc="lower center", fontsize=14)
#legend.get_frame().set_facecolor('#efefef66')
#legend.set_zorder(100)

for i in range(3):
    for j in range(1):
        ax[i,j].xaxis.set_tick_params(labelsize=18)
        ax[i,j].yaxis.set_tick_params(labelsize=18)


#plt.tight_layout()
plt.savefig(os.path.join(path_out, "roc_analysis_zscore_only.jpg"), dpi=300)