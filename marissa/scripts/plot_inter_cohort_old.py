from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
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
# CONFIDENCE INTERVAL ##################################################################################################
########################################################################################################################
alpha = 0.05

ref_str = "##".join([reference_str[ii] for ii in range(len(reference_str))])
train_str = ["##".join(info_data[ii,:].tolist()) for ii in range(len(info_data))]
indeces = np.argwhere(np.array(train_str)==ref_str).flatten()
CIs = {}

reft1 = []
for idx in indeces:
    dcm = project.get_data(data[idx][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data[idx][1]))[0][0]
    md = project.get_standardization(dcm, mask, setupID, True)
    reft1.append(np.mean(md.value_progression[0]))

cil, cih = tool_statistics.get_confidence_interval(reft1, alpha)
CIs["Reference"] = [cil, cih]

counter = 0
for d in [data_test, data_patient_hcm, data_patient_amy]:
    t1s = []
    for i in range(len(d)):
        dcm = project.get_data(d[i][0])[0]
        mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(d[i][1]))[0][0]
        md = project.get_standardization(dcm, mask, setupID, True)

        if np.mean(md.value_progression[-1]) < 1000:
            print(str(np.round(np.mean(md.value_progression[0]))) + "ms -> " + str(np.round(np.mean(md.value_progression[-1]))) + "ms @ " + str(dcm[0x0018, 0x0087].value) + "T " + str(dcm.PatientName) + " " + str(dcm[0x0008, 0x0018].value))

        t1s.append(np.mean(md.value_progression[-1]))

    cil, cih = tool_statistics.get_confidence_interval(t1s, alpha)
    CIs[cohorts[counter]] = [cil, cih]
    counter = counter+1

########################################################################################################################
# PROGRESSION & BOX PLOT ###############################################################################################
########################################################################################################################
all_y_healthy = []
all_y_patient_hcm = []
all_y_patient_amy = []

cols = 4
rows = 2

fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True)
subfigs = fig.subfigures(1, 3, width_ratios = [2, 1, 1])


axes_sf0 = subfigs[0].subplots(3,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1] * 3})
axes_sf1 = subfigs[1].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1] * 1})
axes_sf2 = subfigs[2].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1] * 1})
axes = np.hstack((axes_sf0, axes_sf1, axes_sf2))

#subfigs[0].supylabel("native", fontsize=24, fontweight="bold")
#axes[0,0].set_title("refU", fontsize=24, fontweight="bold")

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

        if j == 0:
            axes[0].scatter(j, y, s=3, c="#660066", zorder=10)
        elif j == (len(md.value_progression) -1):
            axes[0].scatter(j, y, s=3, c="#008000", zorder=10)
        else:
            axes[0].scatter(j, y, s=3, c="#0000ff", zorder=10)
    axes[0].plot(np.arange(0, len(md.value_progression)), prog_y, c="gray", ls="--", lw=0.5)
    all_y_healthy.append(prog_y)
axes[0].text(0.5, 0.5, 'H E A L T H Y', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, color="#00800044", fontsize=90)

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

        if j == 0:
            axes[1].scatter(j, y, s=3, c="#660066", zorder=10)
        elif j == (len(md.value_progression) -1):
            axes[1].scatter(j, y, s=3, c="#ff8000", zorder=10)
        else:
            axes[1].scatter(j, y, s=3, c="#0000ff", zorder=10)
    axes[1].plot(np.arange(0, len(md.value_progression)), prog_y, c="gray", ls="--", lw=0.5)
    all_y_patient_hcm.append(prog_y)
axes[1].text(0.5, 0.5, 'H  C  M', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes, color="#ff800044", fontsize=90)

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

        if j == 0:
            axes[2].scatter(j, y, s=3, c="#660066", zorder=10)
        elif j == (len(md.value_progression) -1):
            axes[2].scatter(j, y, s=3, c="#ff0000", zorder=10)
        else:
            axes[2].scatter(j, y, s=3, c="#0000ff", zorder=10)
    axes[2].plot(np.arange(0, len(md.value_progression)), prog_y, c="gray", ls="--", lw=0.5)
    all_y_patient_amy.append(prog_y)
axes[2].text(0.5, 0.5, 'AMYLOIDOSIS', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes, color="#ff000044", fontsize=90)


# PLOT 1, 2, 3 GENERAL
all_y = all_y_healthy + all_y_patient_hcm + all_y_patient_amy

#dy = (np.max(all_y) - np.min(all_y)) / 100
dy = 50
for i in range(3):
    r1 = plt.Rectangle((-0.25, np.min(all_y) - dy), (len(md.value_progression)-0.5), (np.max(all_y) - np.min(all_y) + 2 * dy), fill=True, color=colours[i] + "11", zorder=1)
    axes[i].add_patch(r1)

    axes[i].grid(lw=1, c="#ffffff")
    axes[i].set_xlim((-0.25, len(md.value_progression) - 0.75))
    axes[i].set_ylim((np.min(all_y) - dy, np.max(all_y) + dy))
    axes[i].set_ylabel("mean T1 [ms]", fontsize=11)

    axes[i].set_xticks([0, len(md.value_progression)-1], ["original\nvalues", "standardized\nvalues"], fontsize=11)

    counter = 0
    for tl in axes[i].get_xticklabels():
        if counter == 0:
            tl.set_color("#660066")
        else:
            tl.set_color(colours[i])
        counter = counter+1

    if len(md.value_progression) > 2:
        axes[i].set_xticks(np.arange(1, len(md.value_progression)).astype(int) - 0.5, labels=parameter_name, minor=True, fontsize=10, c="gray")

# PLOT 4: BOXPLOT
axes[3].boxplot(np.array(all_y_healthy)[:,-1].flatten(), positions=[0.5], widths=0.5, zorder=10, flierprops=dict(markerfacecolor=colours[0], markeredgecolor=colours[0], marker='.', markersize=5), boxprops=dict(color=colours[0]), capprops=dict(color=colours[0]), whiskerprops=dict(color=colours[0]), medianprops=dict(color=colours[0]+"44", lw=2))
axes[3].boxplot(np.array(all_y_patient_hcm)[:,-1].flatten(), positions=[1.5], widths=0.5, zorder=10, flierprops=dict(markerfacecolor=colours[1], markeredgecolor=colours[1], marker='.', markersize=5), boxprops=dict(color=colours[1]), capprops=dict(color=colours[1]), whiskerprops=dict(color=colours[1]), medianprops=dict(color=colours[1]+"44", lw=2))
axes[3].boxplot(np.array(all_y_patient_amy)[:,-1].flatten(), positions=[2.5], widths=0.5, zorder=10, flierprops=dict(markerfacecolor=colours[2], markeredgecolor=colours[2], marker='.', markersize=5), boxprops=dict(color=colours[2]), capprops=dict(color=colours[2]), whiskerprops=dict(color=colours[2]), medianprops=dict(color=colours[2]+"44", lw=2))

for i in range(3):
    r1 = plt.Rectangle((i, np.min(np.array(all_y)[:,-1] - dy)), 1, np.max(np.array(all_y)[:,-1] + dy), fill=True, color=colours[i] + "11", zorder=1)
    axes[3].add_patch(r1)

axes[3].plot([0.5, 0.5, 1.48, 1.48], [np.max(np.array(all_y)[:,-1])+10, np.max(np.array(all_y)[:,-1])+20, np.max(np.array(all_y)[:,-1])+20, np.max(np.array(all_y)[:,-1])+10], c="gray")
axes[3].plot([1.52, 1.52, 2.5, 2.5], [np.max(np.array(all_y)[:,-1])+10, np.max(np.array(all_y)[:,-1])+20, np.max(np.array(all_y)[:,-1])+20, np.max(np.array(all_y)[:,-1])+10], c="gray")
axes[3].plot([0.5, 0.5, 2.5, 2.5], [np.max(np.array(all_y)[:,-1])+30, np.max(np.array(all_y)[:,-1])+40, np.max(np.array(all_y)[:,-1])+40, np.max(np.array(all_y)[:,-1])+30], c="gray")

ph = stats.shapiro(np.array(all_y_healthy)[:,-1])[-1]
phcm = stats.shapiro(np.array(all_y_patient_hcm)[:,-1])[-1]
pamy = stats.shapiro(np.array(all_y_patient_amy)[:,-1])[-1]

if ph < 0.05 or phcm < 0.05:
    statistic = stats.mannwhitneyu(np.array(all_y_healthy)[:,-1], np.array(all_y_patient_hcm)[:,-1])
else:
    statistic = stats.ttest_ind(np.array(all_y_healthy)[:,-1], np.array(all_y_patient_hcm)[:,-1])
phhcm = statistic.pvalue

if ph < 0.05 or pamy < 0.05:
    statistic = stats.mannwhitneyu(np.array(all_y_healthy)[:,-1], np.array(all_y_patient_amy)[:,-1])
else:
    statistic = stats.ttest_ind(np.array(all_y_healthy)[:,-1], np.array(all_y_patient_amy)[:,-1])
phamy = statistic.pvalue

if pamy < 0.05 or phcm < 0.05:
    statistic = stats.mannwhitneyu(np.array(all_y_patient_amy)[:,-1], np.array(all_y_patient_hcm)[:,-1])
else:
    statistic = stats.ttest_ind(np.array(all_y_patient_amy)[:,-1], np.array(all_y_patient_hcm)[:,-1])
pamyhcm = statistic.pvalue


axes[3].text(1.0, np.max(np.array(all_y)[:,-1])+25, ("*" if phhcm < 0.05 else "n.s."), horizontalalignment='center', verticalalignment='center', color="gray", fontsize=12)
axes[3].text(2.0, np.max(np.array(all_y)[:,-1])+25, ("*" if pamyhcm < 0.05 else "n.s."), horizontalalignment='center', verticalalignment='center', color="gray", fontsize=12)
axes[3].text(1.5, np.max(np.array(all_y)[:,-1])+45, ("*" if phamy < 0.05 else "n.s."), horizontalalignment='center', verticalalignment='center', color="gray", fontsize=12)

axes[3].set_xlim([0, 3])
axes[3].set_xticks([0.5, 1.5, 2.5], ["Healthy\n", "HCM\n", "Amyloidosis\n"], fontsize=11)
counter = 0
for tl in axes[3].get_xticklabels():
    tl.set_color(colours[counter])
    counter = counter+1

miny = np.min(np.array(all_y)[:,-1])-dy
maxy = np.max(np.array(all_y)[:,-1])+2*dy

axes[3].set_ylim([miny, maxy])
axes[3].set_ylabel("mean T1 [ms]", fontsize=11)

# CONFIDENCE INTERVAL
r1 = plt.Rectangle((0, CIs["Reference"][0]), 3, CIs["Reference"][1]-CIs["Reference"][0], fill=True, color="#66006644", zorder=2)
axes[4].add_patch(r1)
axes[4].axhline(CIs["Reference"][0], c="#660066", lw=1, linestyle=":")
axes[4].axhline(CIs["Reference"][1], c="#660066", lw=1, linestyle=":")

for i in range(3):
    r1 = plt.Rectangle((i, miny), 1, maxy-miny, fill=True, color=colours[i] + "11", zorder=1)
    axes[4].add_patch(r1)

    ci = CIs[cohorts[i]]
    axes[4].plot([i+0.5, i+0.5], ci, c=colours[i])
    axes[4].plot([i+0.45, i+0.55], [ci[0], ci[0]], c=colours[i])
    axes[4].plot([i+0.45, i+0.55], [ci[1], ci[1]], c=colours[i])
    axes[4].scatter([i+0.5], [np.mean(ci)], c=colours[i], zorder=3)

axes[4].set_xlim((0, 3))
axes[4].set_ylim((miny, maxy))

axes[4].set_xticks([0.5, 1.5, 2.5], ["Healthy", "HCM", "Amyloidosis"], fontsize=11)
counter = 0
for tl in axes[4].get_xticklabels():
    tl.set_color(colours[counter])
    counter = counter+1

axes[4].set_ylabel("mean T1 [ms]")


# SAVE PLOT
plt.gcf().set_dpi(600)
plt.savefig(os.path.join(path_out, "inter_cohort_progression.jpg"), dpi=600)

########################################################################################################################
# STATISTICAL ANALYSIS #################################################################################################
########################################################################################################################

healthy_before = np.array(all_y_healthy)[:,0].flatten()
healthy_after = np.array(all_y_healthy)[:,-1].flatten()
patient_before_hcm = np.array(all_y_patient_hcm)[:,0].flatten()
patient_after_hcm = np.array(all_y_patient_hcm)[:,-1].flatten()
patient_before_amy = np.array(all_y_patient_amy)[:,0].flatten()
patient_after_amy = np.array(all_y_patient_amy)[:,-1].flatten()

file_info = os.path.join(path_out, "inter_cohort_statistical_analysis.txt")
if os.path.exists(file_info):
    os.remove(file_info)

with open(file_info, "a") as file:
    file.write("BEFORE")
    file.write("\nHealthy:\t" + str(np.round(np.mean(healthy_before),2)) + " +/- " + str(np.round(np.std(healthy_before),2)) + " [" + str(np.round(np.min(healthy_before),2)) + "-" + str(np.round(np.max(healthy_before),2)) + "] | CoV[%] = " + str(np.round(100 * np.std(healthy_before) / np.mean(healthy_before),2)))
    statistic = stats.shapiro(healthy_before)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient HCM:\t" + str(np.round(np.mean(patient_before_hcm),2)) + " +/- " + str(np.round(np.std(patient_before_hcm),2)) + " [" + str(np.round(np.min(patient_before_hcm),2)) + "-" + str(np.round(np.max(patient_before_hcm),2)) + "] | CoV[%] = " + str(np.round(100 * np.std(patient_before_hcm) / np.mean(patient_before_hcm),2)))
    statistic = stats.shapiro(patient_before_hcm)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient AMY:\t" + str(np.round(np.mean(patient_before_amy),2)) + " +/- " + str(np.round(np.std(patient_before_amy),2)) + " [" + str(np.round(np.min(patient_before_amy),2)) + "-" + str(np.round(np.max(patient_before_amy),2)) + "] | CoV[%] = " + str(np.round(100 * np.std(patient_before_amy) / np.mean(patient_before_amy),2)))
    statistic = stats.shapiro(patient_before_amy)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\n")

    statistic = stats.ttest_ind(healthy_before, patient_before_hcm)
    file.write("\nHealthy vs. Patient HCM ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.mannwhitneyu(healthy_before, patient_before_hcm)
    file.write("\nHealthy vs. Patient HCM Mann-Whitney-U p = " + str(np.round(statistic.pvalue, 3)))

    statistic = stats.ttest_ind(healthy_before, patient_before_amy)
    file.write("\nHealthy vs. Patient AMY ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.mannwhitneyu(healthy_before, patient_before_amy)
    file.write("\nHealthy vs. Patient AMY Mann-Whitney-U p = " + str(np.round(statistic.pvalue, 3)))

    statistic = stats.ttest_ind(patient_before_hcm, patient_before_amy)
    file.write("\nPatient HCM vs. Patient AMY ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.mannwhitneyu(patient_before_hcm, patient_before_amy)
    file.write("\nPatient HCM vs. Patient AMY Mann-Whitney-U p = " + str(np.round(statistic.pvalue, 3)))

    file.write("\n")

    statistic = stats.f_oneway(healthy_before, patient_before_hcm, patient_before_amy)
    file.write("\nANOVA p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.kruskal(healthy_before, patient_before_hcm, patient_before_amy)
    file.write("\nKruskal-Wallis p = " + str(np.round(statistic.pvalue, 3)))

    file.write("\n")

    roc_analysis = tool_statistics.roc_curve_analysis(patient_before_hcm, healthy_before)
    file.write("\nHealthy vs Patient HCM ROC Analysis:")
    file.write("\n\tOptimal Threshhold = " + str(roc_analysis["optimal_threshhold"]))
    file.write("\n\tSensitivity = " + str(roc_analysis["optimal_confusion_matrix"]["TPR"]))
    file.write("\n\tSpecificity = " + str(roc_analysis["optimal_confusion_matrix"]["TNR"]))
    file.write("\n\tAccuracy = " + str(roc_analysis["optimal_confusion_matrix"]["ACC"]))

    roc_analysis = tool_statistics.roc_curve_analysis(patient_before_amy, healthy_before)
    file.write("\nHealthy vs Patient AMY ROC Analysis:")
    file.write("\n\tOptimal Threshhold = " + str(roc_analysis["optimal_threshhold"]))
    file.write("\n\tSensitivity = " + str(roc_analysis["optimal_confusion_matrix"]["TPR"]))
    file.write("\n\tSpecificity = " + str(roc_analysis["optimal_confusion_matrix"]["TNR"]))
    file.write("\n\tAccuracy = " + str(roc_analysis["optimal_confusion_matrix"]["ACC"]))

    roc_analysis = tool_statistics.roc_curve_analysis(patient_before_amy, patient_before_hcm)
    file.write("\nPatient HCM vs Patient AMY ROC Analysis:")
    file.write("\n\tOptimal Threshhold = " + str(roc_analysis["optimal_threshhold"]))
    file.write("\n\tSensitivity = " + str(roc_analysis["optimal_confusion_matrix"]["TPR"]))
    file.write("\n\tSpecificity = " + str(roc_analysis["optimal_confusion_matrix"]["TNR"]))
    file.write("\n\tAccuracy = " + str(roc_analysis["optimal_confusion_matrix"]["ACC"]))


    file.write("\n\nAFTER")
    file.write("\nHealthy:\t" + str(np.round(np.mean(healthy_after),2)) + " +/- " + str(np.round(np.std(healthy_after),2)) + " [" + str(np.round(np.min(healthy_after),2)) + "-" + str(np.round(np.max(healthy_after),2)) + "] | CoV[%] = " + str(np.round(100 * np.std(healthy_after) / np.mean(healthy_after),2)))
    statistic = stats.shapiro(healthy_after)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient HCM:\t" + str(np.round(np.mean(patient_after_hcm),2)) + " +/- " + str(np.round(np.std(patient_after_hcm),2)) + " [" + str(np.round(np.min(patient_after_hcm),2)) + "-" + str(np.round(np.max(patient_after_hcm),2)) + "] | CoV[%] = " + str(np.round(100 * np.std(patient_after_hcm) / np.mean(patient_after_hcm),2)))
    statistic = stats.shapiro(patient_after_hcm)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient AMY:\t" + str(np.round(np.mean(patient_after_amy),2)) + " +/- " + str(np.round(np.std(patient_after_amy),2)) + " [" + str(np.round(np.min(patient_after_amy),2)) + "-" + str(np.round(np.max(patient_after_amy),2)) + "] | CoV[%] = " + str(np.round(100 * np.std(patient_after_amy) / np.mean(patient_after_amy),2)))
    statistic = stats.shapiro(patient_after_amy)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\n")

    statistic = stats.ttest_ind(healthy_after, patient_after_hcm)
    file.write("\nHealthy vs. Patient HCM ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.mannwhitneyu(healthy_after, patient_after_hcm)
    file.write("\nHealthy vs. Patient HCM Mann-Whitney-U p = " + str(np.round(statistic.pvalue, 3)))

    statistic = stats.ttest_ind(healthy_after, patient_after_amy)
    file.write("\nHealthy vs. Patient AMY ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.mannwhitneyu(healthy_after, patient_after_amy)
    file.write("\nHealthy vs. Patient AMY Mann-Whitney-U p = " + str(np.round(statistic.pvalue, 3)))

    statistic = stats.ttest_ind(patient_after_hcm, patient_after_amy)
    file.write("\nPatient HCM vs. Patient AMY ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.mannwhitneyu(patient_after_hcm, patient_after_amy)
    file.write("\nPatient HCM vs. Patient AMY Mann-Whitney-U p = " + str(np.round(statistic.pvalue, 3)))

    file.write("\n")

    statistic = stats.f_oneway(healthy_after, patient_after_hcm, patient_after_amy)
    file.write("\nANOVA p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.kruskal(healthy_after, patient_after_hcm, patient_after_amy)
    file.write("\nKruskal-Wallis p = " + str(np.round(statistic.pvalue, 3)))


    file.write("\n")

    roc_analysis = tool_statistics.roc_curve_analysis(patient_after_hcm, healthy_after)
    file.write("\nHealthy vs Patient HCM ROC Analysis:")
    file.write("\n\tOptimal Threshhold = " + str(roc_analysis["optimal_threshhold"]))
    file.write("\n\tSensitivity = " + str(roc_analysis["optimal_confusion_matrix"]["TPR"]))
    file.write("\n\tSpecificity = " + str(roc_analysis["optimal_confusion_matrix"]["TNR"]))
    file.write("\n\tAccuracy = " + str(roc_analysis["optimal_confusion_matrix"]["ACC"]))

    roc_analysis = tool_statistics.roc_curve_analysis(patient_after_amy, healthy_after)
    file.write("\nHealthy vs Patient AMY ROC Analysis:")
    file.write("\n\tOptimal Threshhold = " + str(roc_analysis["optimal_threshhold"]))
    file.write("\n\tSensitivity = " + str(roc_analysis["optimal_confusion_matrix"]["TPR"]))
    file.write("\n\tSpecificity = " + str(roc_analysis["optimal_confusion_matrix"]["TNR"]))
    file.write("\n\tAccuracy = " + str(roc_analysis["optimal_confusion_matrix"]["ACC"]))

    roc_analysis = tool_statistics.roc_curve_analysis(patient_after_amy, patient_after_hcm)
    file.write("\nPatient HCM vs Patient AMY ROC Analysis:")
    file.write("\n\tOptimal Threshhold = " + str(roc_analysis["optimal_threshhold"]))
    file.write("\n\tSensitivity = " + str(roc_analysis["optimal_confusion_matrix"]["TPR"]))
    file.write("\n\tSpecificity = " + str(roc_analysis["optimal_confusion_matrix"]["TNR"]))
    file.write("\n\tAccuracy = " + str(roc_analysis["optimal_confusion_matrix"]["ACC"]))

    file.write("\n")

    file.write("\nConfidence Intervals")
    file.write("\nHealthy: " + str(np.round(CIs["Healthy"][0],2)) +  " - " + str(np.round(CIs["Healthy"][1],2)))
    file.write("\nHCM: " + str(np.round(CIs["HCM"][0],2)) +  " - " + str(np.round(CIs["HCM"][1],2)))
    file.write("\nAmyloidosis: " + str(np.round(CIs["Amyloidosis"][0],2)) +  " - " + str(np.round(CIs["Amyloidosis"][1],2)))

    file.close()
