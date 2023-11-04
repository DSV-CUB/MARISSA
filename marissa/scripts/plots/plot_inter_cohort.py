from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
from marissa.scripts.plots import basic_functions
import numpy as np
import os
from matplotlib import pyplot as plt
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
parameter_name = ["age", "sex", "scanner", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]
reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]

colours = ["#008000", "#ff8000", "#ff0000"]
cohorts = ["HTE", "HCM", "AMY"]

alpha_CI = 0.05

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

########################################################################################################################
# CONFIDENCE INTERVAL ##################################################################################################
########################################################################################################################
CI_reference = []
CIs_after = {}
CIs_before = {}

# CI of data that captures reference CP environment
ref_str = "##".join([reference_str[ii] for ii in range(len(reference_str))])
train_str = ["##".join(info_data[ii,:].tolist()) for ii in range(len(info_data))]
indeces = np.argwhere(np.array(train_str)==ref_str).flatten()
reft1 = []
for idx in indeces:
    dcm = project.get_data(data[idx][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data[idx][1]))[0][0]
    md = project.get_standardization(dcm, mask, setupID, True)
    reft1.append(np.mean(md.value_progression[0]))
cil, cih = tool_statistics.get_confidence_interval(reft1, alpha_CI)
CI_reference = [cil, cih]

# CI before and after standardization for each cohort
counter = 0
for d in [data_test, data_patient_hcm, data_patient_amy]:
    t1s_after = []
    t1s_before = []

    for i in range(len(d)):
        dcm = project.get_data(d[i][0])[0]
        mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(d[i][1]))[0][0]
        md = project.get_standardization(dcm, mask, setupID, True)
        t1s_after.append(np.mean(md.value_progression[-1]))
        t1s_before.append(np.mean(md.value_progression[0]))
        #if np.mean(md.value_progression[-1]) < 1000:
        #    print("Standardized to 3T below 1000ms: " + str(np.round(np.mean(md.value_progression[0]))) + "ms -> " + str(np.round(np.mean(md.value_progression[-1]))) + "ms @ " + str(dcm[0x0018, 0x0087].value) + "T " + str(dcm.PatientName) + " " + str(dcm[0x0008, 0x0018].value))

    cil, cih = tool_statistics.get_confidence_interval(t1s_after, alpha_CI)
    CIs_after[cohorts[counter]] = [cil, cih]

    cil, cih = tool_statistics.get_confidence_interval(t1s_before, alpha_CI)
    CIs_before[cohorts[counter]] = [cil, cih]

    counter = counter+1

########################################################################################################################
# INTER-COHORT PLOT ####################################################################################################
########################################################################################################################
# General
all_y_healthy = []
all_y_patient_hcm = []
all_y_patient_amy = []

cols = 6
rows = 3

fig = plt.figure(None, figsize=(cols*5, rows*5), constrained_layout=True)
subfigs = fig.subfigures(1, 3, width_ratios = [1,1.5,1])

axes_sf0 = subfigs[0].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1] * 1}) #0
#axes_sf1 = subfigs[1].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1] * 1}) #1
axes_sf2 = subfigs[1].subplots(3,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1] * 3}) #2,3,4
#axes_sf3 = subfigs[3].subplots(1,1, gridspec_kw={'width_ratios': [1] * 1, 'height_ratios': [1] * 1}) #5
axes_sf4 = subfigs[2].subplots(1,2, gridspec_kw={'width_ratios': [1] * 2, 'height_ratios': [1] * 1}) #6

#axes = np.hstack((axes_sf0, axes_sf1, axes_sf2, axes_sf3, axes_sf4))
axes = np.hstack((axes_sf0, axes_sf2, axes_sf4))

subfigs[0].suptitle("before standardization", fontsize=26, fontweight="bold")
subfigs[1].suptitle("\u2192     BPSP     \u2192", fontsize=26, fontweight="bold")
subfigs[2].suptitle("after standardization", fontsize=26, fontweight="bold")

# PLOT 2, 3, 4 (MIDDLE PART --> INTRA-COHORT VALUE PROGRESSION FOR EACH CONFOUNDING PARAMETER)
# PLOT 2: Healthy Volunteers
def plot_progression(project, data, ax, c_out):
    prog_list = []
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

            if j == 0:
                ax.scatter(j, y, s=5, c="#660066", zorder=10)
            elif j == (len(md.value_progression) -1):
                ax.scatter(j, y, s=5, c=c_out, zorder=10)
            else:
                ax.scatter(j, y, s=5, c="#0000ff", zorder=10)
        ax.plot(np.arange(0, len(md.value_progression)), prog_y, c="gray", ls="--", lw=0.5)
        prog_list.append(prog_y)
    return prog_list

all_y_healthy = plot_progression(project, data_test, axes[2], "#008000")
axes[2].text(0.5, 0.5, 'H  T  E', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes, color="#00800044", fontsize=200)

all_y_patient_hcm = plot_progression(project, data_patient_hcm, axes[3], "#ff8000")
axes[3].text(0.5, 0.5, 'H  C  M', horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes, color="#ff800044", fontsize=200)

all_y_patient_amy = plot_progression(project, data_patient_amy, axes[4], "#ff0000")
axes[4].text(0.5, 0.5, 'A  M  Y', horizontalalignment='center', verticalalignment='center', transform=axes[4].transAxes, color="#ff000044", fontsize=200)


# PLOT 2, 3, 4 GENERAL FORMAT
all_y = all_y_healthy + all_y_patient_hcm + all_y_patient_amy
dy=50
miny=np.min(all_y) - dy
maxy = np.max(all_y) + 1.5*dy

for ii in range(3):
    i = ii + 2
    r1 = plt.Rectangle((-0.25, miny), (len(md.value_progression)-0.5), maxy, fill=True, color=colours[ii] + "11", zorder=1)
    axes[i].add_patch(r1)

    axes[i].grid(lw=1, c="#ffffff")
    axes[i].set_xlim((-0.25, len(md.value_progression) - 0.75))
    axes[i].set_ylim((miny, maxy))
    axes[i].set_ylabel("mean T1 [ms]", fontsize=22)

    axes[i].set_xticks([0, len(md.value_progression)-1], ["original\nvalues", "standardized\nvalues"], fontsize=22)

    counter = 0
    for tl in axes[i].get_xticklabels():
        if counter == 0:
            tl.set_color("#660066")
        else:
            tl.set_color(colours[ii])
        counter = counter+1

    if len(md.value_progression) > 2:
        axes[i].set_xticks(np.arange(1, len(md.value_progression)).astype(int) - 0.5, labels=parameter_name, minor=True, fontsize=18, c="#444444", rotation=45, ha='right', rotation_mode='anchor')

    axes[i].yaxis.set_tick_params(labelsize=20)

# PLOT 0, 5: BOXPLOT
def plot_boxplot(healthy, hcm, amy, ax, colours, lower_bound, upper_bound, max_y):
    ax.boxplot(healthy, positions=[0.5], widths=0.5, zorder=10, flierprops=dict(markerfacecolor=colours[0], markeredgecolor=colours[0], marker='.', markersize=5), boxprops=dict(color=colours[0]), capprops=dict(color=colours[0]), whiskerprops=dict(color=colours[0]), medianprops=dict(color=colours[0]+"44", lw=2))
    ax.boxplot(hcm, positions=[1.5], widths=0.5, zorder=10, flierprops=dict(markerfacecolor=colours[1], markeredgecolor=colours[1], marker='.', markersize=5), boxprops=dict(color=colours[1]), capprops=dict(color=colours[1]), whiskerprops=dict(color=colours[1]), medianprops=dict(color=colours[1]+"44", lw=2))
    ax.boxplot(amy, positions=[2.5], widths=0.5, zorder=10, flierprops=dict(markerfacecolor=colours[2], markeredgecolor=colours[2], marker='.', markersize=5), boxprops=dict(color=colours[2]), capprops=dict(color=colours[2]), whiskerprops=dict(color=colours[2]), medianprops=dict(color=colours[2]+"44", lw=2))

    for i in range(3):
        r1 = plt.Rectangle((i, lower_bound), 1, upper_bound, fill=True, color=colours[i] + "11", zorder=1)
        ax.add_patch(r1)

    ax.plot([0.5, 0.5, 1.48, 1.48], [max_y+10, max_y+20, max_y+20, max_y+10], c="#444444")
    ax.plot([1.52, 1.52, 2.5, 2.5], [max_y+10, max_y+20, max_y+20, max_y+10], c="#444444")
    ax.plot([0.5, 0.5, 2.5, 2.5], [max_y+30, max_y+40, max_y+40, max_y+30], c="#444444")

    ph = stats.shapiro(healthy)[-1]
    phcm = stats.shapiro(hcm)[-1]
    pamy = stats.shapiro(amy)[-1]

    if ph < 0.05 or phcm < 0.05 or pamy < 0.05:
        statistic = stats.kruskal(healthy, hcm, amy)
        pall = statistic.pvalue
        statistic = stats.mannwhitneyu(healthy, hcm)
        phhcm = statistic.pvalue
        statistic = stats.mannwhitneyu(healthy, amy)
        phamy = statistic.pvalue
        statistic = stats.mannwhitneyu(amy, hcm)
        pamyhcm = statistic.pvalue
    else:
        statistic = stats.f_oneway(healthy, hcm, amy)
        pall = statistic.pvalue
        statistic = stats.ttest_ind(healthy, hcm)
        phhcm = statistic.pvalue
        statistic = stats.ttest_ind(healthy, amy)
        phamy = statistic.pvalue
        statistic = stats.ttest_ind(amy, hcm)
        pamyhcm = statistic.pvalue

    ax.text(1.0, max_y+25, ("*" if (phhcm < 0.05 and pall < 0.05) else "n.s."), horizontalalignment='center', verticalalignment='center', color="#444444", fontsize=20)
    ax.text(2.0, max_y+25, ("*" if (pamyhcm < 0.05 and pall < 0.05) else "n.s."), horizontalalignment='center', verticalalignment='center', color="#444444", fontsize=20)
    ax.text(1.5, max_y+45, ("*" if (phamy < 0.05 and pall < 0.05) else "n.s."), horizontalalignment='center', verticalalignment='center', color="#444444", fontsize=20)

    ax.set_xlim([0, 3])
    ax.set_xticks([0.5, 1.5, 2.5], ["HTE\n", "HCM\n", "AMY\n"], fontsize=22)
    counter = 0
    for tl in ax.get_xticklabels():
        tl.set_color(colours[counter])
        counter = counter+1

    ax.set_ylim([lower_bound, upper_bound])
    ax.set_ylabel("mean T1 [ms]", fontsize=22)
    ax.yaxis.set_tick_params(labelsize=20)
    return

# before standardization
plot_boxplot(np.array(all_y_healthy)[:,0].flatten(), np.array(all_y_patient_hcm)[:,0].flatten(), np.array(all_y_patient_amy)[:,0].flatten(), axes[0], colours, miny, maxy, np.max(np.array(all_y)))
# after standardization
plot_boxplot(np.array(all_y_healthy)[:,-1].flatten(), np.array(all_y_patient_hcm)[:,-1].flatten(), np.array(all_y_patient_amy)[:,-1].flatten(), axes[5], colours, miny, maxy, np.max(np.array(all_y)))


# PLOT 1, 6: CONFIDENCE INTERVAL
def plot_confidence_interval(CIref, CI, ax, lower_bound, upper_bound, colours):
    # plot reference ci as rectangle
    r1 = plt.Rectangle((0, CIref[0]), 3, CIref[1]-CIref[0], fill=True, color="#66006644", zorder=2)
    ax.add_patch(r1)
    ax.axhline(CIref[0], c="#660066", lw=1, linestyle=":")
    ax.axhline(CIref[1], c="#660066", lw=1, linestyle=":")

    cohorts = list(CI.keys())

    for i in range(len(cohorts)):
        r1 = plt.Rectangle((i, lower_bound), 1, upper_bound-lower_bound, fill=True, color=colours[i] + "11", zorder=1)
        ax.add_patch(r1)

        ci = CI[cohorts[i]]
        ax.plot([i+0.5, i+0.5], ci, c=colours[i])
        ax.plot([i+0.40, i+0.60], [ci[0], ci[0]], c=colours[i])
        ax.plot([i+0.40, i+0.60], [ci[1], ci[1]], c=colours[i])
        ax.scatter([i+0.5], [np.mean(ci)], c=colours[i], zorder=3, s=30)

    ax.set_xlim((0, len(cohorts)))
    ax.set_ylim((miny, maxy))

    ax.set_xticks([0.5, 1.5, 2.5], ["HTE\n", "HCM\n", "AMY\n"], fontsize=22)
    counter = 0
    for tl in ax.get_xticklabels():
        tl.set_color(colours[counter])
        counter = counter+1

    ax.set_ylabel("mean T1 [ms]", fontsize=22)
    ax.yaxis.set_tick_params(labelsize=20)
    return

plot_confidence_interval(CI_reference, CIs_before, axes[1], miny, maxy, colours)
plot_confidence_interval(CI_reference, CIs_after, axes[6], miny, maxy, colours)

# SAVE PLOT
#plt.tight_layout()
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
    file.write("\nHealthy:\t" + str(np.round(np.mean(healthy_before),2)) + " +/- " + str(np.round(np.std(healthy_before),2)) + " [" + str(np.round(np.min(healthy_before),2)) + "-" + str(np.round(np.max(healthy_before),2)) + "] 25/75 Quantiles:" + str(np.round(np.quantile(healthy_before, 0.25),2)) + " +/- " + str(np.round(np.quantile(healthy_before, 0.75),2)) + " | CoV[%] = " + str(np.round(100 * np.std(healthy_before) / np.mean(healthy_before),2)))
    statistic = stats.shapiro(healthy_before)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient HCM:\t" + str(np.round(np.mean(patient_before_hcm),2)) + " +/- " + str(np.round(np.std(patient_before_hcm),2)) + " [" + str(np.round(np.min(patient_before_hcm),2)) + "-" + str(np.round(np.max(patient_before_hcm),2)) + "] 25/75 Quantiles:" + str(np.round(np.quantile(patient_before_hcm, 0.25),2)) + " +/- " + str(np.round(np.quantile(patient_before_hcm, 0.75),2)) + " | CoV[%] = " + str(np.round(100 * np.std(patient_before_hcm) / np.mean(patient_before_hcm),2)))
    statistic = stats.shapiro(patient_before_hcm)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient AMY:\t" + str(np.round(np.mean(patient_before_amy),2)) + " +/- " + str(np.round(np.std(patient_before_amy),2)) + " [" + str(np.round(np.min(patient_before_amy),2)) + "-" + str(np.round(np.max(patient_before_amy),2)) + "] 25/75 Quantiles:" + str(np.round(np.quantile(patient_before_amy, 0.25),2)) + " +/- " + str(np.round(np.quantile(patient_before_amy, 0.75),2)) + " | CoV[%] = " + str(np.round(100 * np.std(patient_before_amy) / np.mean(patient_before_amy),2)))
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

    file.write("\n")

    file.write("\nConfidence Intervals")
    file.write("\nHealthy: " + str(np.round(CIs_before["HTE"][0],2)) +  " - " + str(np.round(CIs_before["HTE"][1],2)))
    file.write("\nHCM: " + str(np.round(CIs_before["HCM"][0],2)) +  " - " + str(np.round(CIs_before["HCM"][1],2)))
    file.write("\nAmyloidosis: " + str(np.round(CIs_before["AMY"][0],2)) +  " - " + str(np.round(CIs_before["AMY"][1],2)))

    file.write("\n\nAFTER")
    file.write("\nHealthy:\t" + str(np.round(np.mean(healthy_after),2)) + " +/- " + str(np.round(np.std(healthy_after),2)) + " [" + str(np.round(np.min(healthy_after),2)) + "-" + str(np.round(np.max(healthy_after),2)) + "] 25/75 Quantiles:" + str(np.round(np.quantile(healthy_after, 0.25),2)) + " +/- " + str(np.round(np.quantile(healthy_after, 0.75),2)) + " | CoV[%] = " + str(np.round(100 * np.std(healthy_after) / np.mean(healthy_after),2)))
    statistic = stats.shapiro(healthy_after)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient HCM:\t" + str(np.round(np.mean(patient_after_hcm),2)) + " +/- " + str(np.round(np.std(patient_after_hcm),2)) + " [" + str(np.round(np.min(patient_after_hcm),2)) + "-" + str(np.round(np.max(patient_after_hcm),2)) + "] 25/75 Quantiles:" + str(np.round(np.quantile(patient_after_hcm, 0.25),2)) + " +/- " + str(np.round(np.quantile(patient_after_hcm, 0.75),2)) + " | CoV[%] = " + str(np.round(100 * np.std(patient_after_hcm) / np.mean(patient_after_hcm),2)))
    statistic = stats.shapiro(patient_after_hcm)
    file.write("\nShapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))

    file.write("\nPatient AMY:\t" + str(np.round(np.mean(patient_after_amy),2)) + " +/- " + str(np.round(np.std(patient_after_amy),2)) + " [" + str(np.round(np.min(patient_after_amy),2)) + "-" + str(np.round(np.max(patient_after_amy),2)) + "] 25/75 Quantiles:" + str(np.round(np.quantile(patient_after_amy, 0.25),2)) + " +/- " + str(np.round(np.quantile(patient_after_amy, 0.75),2)) + " | CoV[%] = " + str(np.round(100 * np.std(patient_after_amy) / np.mean(patient_after_amy),2)))
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
    file.write("\nHealthy: " + str(np.round(CIs_after["HTE"][0],2)) +  " - " + str(np.round(CIs_after["HTE"][1],2)))
    file.write("\nHCM: " + str(np.round(CIs_after["HCM"][0],2)) +  " - " + str(np.round(CIs_after["HCM"][1],2)))
    file.write("\nAmyloidosis: " + str(np.round(CIs_after["AMY"][0],2)) +  " - " + str(np.round(CIs_after["AMY"][1],2)))

    file.close()
