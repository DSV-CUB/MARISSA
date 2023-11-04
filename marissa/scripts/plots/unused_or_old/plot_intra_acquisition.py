from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# SETUP
project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\DoktorarbeitDSV.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER"
#parameters = ["PatientsAge", "PatientsSex", "BMIgroup3", "System", "T1Sequence"]
#reference = ["18Y", "M", 0, "SIEMENS#Verio#syngo MR B17", "MOLLI"]

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 533"]
reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 533"]

# PREPARATION
project = marissadb.Module(project_path)

pids = []
for pdescr in parameters:
    pids.append(project.select("SELECT parameterID FROM tbl_parameter WHERE description = '" + pdescr + "'")[0][0])

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
    #dcm = project.get_data(selection[i][0])[0]
    #if "d13b" in str(dcm[0x0018, 0x1020].value).lower():
    #    continue
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
# VIOLIN PLOT ##########################################################################################################
########################################################################################################################

fig, ax = plt.subplots()

# HEALTHY TEST GROUP
covs_before = []
covs_after = []
cohort_size = []
cohorts = ["Healthy", "HCM", "Amyloidosis"]
for d in [data_test, data_patient_hcm, data_patient_amy]:
    cohort_size.append(len(d))
    for i in range(len(d)):
        dcm = project.get_data(d[i][0])[0]
        mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(d[i][1]))[0][0]
        md = project.get_standardization(dcm, mask, setupID, True)
        cov_before = 100 * np.std(md.value_progression[0]) / np.mean(md.value_progression[0])
        cov_after = 100 * np.std(md.value_progression[-1]) / np.mean(md.value_progression[-1])

        covs_before.append(cov_before)
        covs_after.append(cov_after)
covs_before = np.array(covs_before)
covs_after = np.array(covs_after)

df = pd.DataFrame()
df["cov"] = np.transpose(np.concatenate((covs_before, covs_after)).astype(float))

group = ["original"] * np.sum(cohort_size) + ["standardized"] * np.sum(cohort_size)
df["group"] = np.array(group)

cohort = []
for _ in range(2):
    for ii in range(3):
        cohort = cohort + [cohorts[ii]] * (cohort_size[ii])
df["cohort"] = np.array(cohort)

sns.violinplot(data=df, x="cohort", y="cov", hue="group", split=True, inner=None, ax=ax, legend=False)

colors_patches = ["#66006644", "#00800044", "#66006644", "#ff800044", "#66006644", "#ff000044"]
counter = 0
for ii in range(len(ax.get_children())):
    if "polycollection" in str(type(ax.get_children()[ii])).lower():
        ax.get_children()[ii].set_color(colors_patches[counter])
        counter = counter + 1

colours = ["#008000", "#ff8000", "#ff0000"]
for i in range(3):
    r1 = plt.Rectangle((i-0.5, 0), 1, np.max(np.concatenate((covs_before, covs_after))) + 2, fill=True, color=colours[i] + "11", zorder=1)
    ax.add_patch(r1)

ax.set_xlim([-0.5, 2.5])
ax.set_xticks([0, 1, 2], ["Healthy", "HCM", "Amyloidosis"], fontsize=11)
colours_xtick = ["#008000", "#ff8000", "#ff0000"]
counter = 0
for tl in ax.get_xticklabels():
    tl.set_color(colours_xtick[counter])
    counter = counter+1

ax.set_xlabel("")
ax.set_ylabel("CoV [%]")
ax.set_ylim([0, np.max(np.concatenate((covs_before, covs_after))) + 2])
plt.legend([], [], frameon=False)

plt.gcf().set_dpi(600)
plt.savefig(os.path.join(path_out, "intra_map_plot.jpg"), dpi=600)


########################################################################################################################
# STATISTICS ###########################################################################################################
########################################################################################################################

file_info = os.path.join(path_out, "intra_map_analysis.txt")
if os.path.exists(file_info):
    os.remove(file_info)

with open(file_info, "a") as file:
    file.write("Healthy - number of maps: " + str(cohort_size[0]))
    cb = covs_before[:cohort_size[0]]
    ca = covs_after[:cohort_size[0]]
    file.write("\nOriginal: " + str(np.round(np.mean(cb), 2)) + "+/-" + str(np.round(np.std(cb), 2)) + " [" + str(np.round(np.min(cb), 2)) + "-" + str(np.round(np.max(cb), 2)) +"]")
    file.write("\nStandardized: " + str(np.round(np.mean(ca), 2)) + "+/-" + str(np.round(np.std(ca), 2)) + " [" + str(np.round(np.min(ca), 2)) + "-" + str(np.round(np.max(ca), 2)) +"]")
    statistic = stats.shapiro(cb)
    file.write("\nOriginal Shapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))
    statistic = stats.shapiro(ca)
    file.write("\nStandardized Shapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))
    statistic = stats.ttest_rel(cb, ca)
    file.write("\nOriginal vs. Standardized ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.wilcoxon(cb, ca)
    file.write("\nOriginal vs. Standardized Wilcoxon p = " + str(np.round(statistic.pvalue, 3)))

    file.write("\n\nHCM - number of maps: " + str(cohort_size[1]))
    cb = covs_before[cohort_size[0]:cohort_size[0]+cohort_size[1]]
    ca = covs_after[cohort_size[0]:cohort_size[0]+cohort_size[1]]
    file.write("\nOriginal: " + str(np.round(np.mean(cb), 2)) + "+/-" + str(np.round(np.std(cb), 2)) + " [" + str(np.round(np.min(cb), 2)) + "-" + str(np.round(np.max(cb), 2)) +"]")
    file.write("\nStandardized: " + str(np.round(np.mean(ca), 2)) + "+/-" + str(np.round(np.std(ca), 2)) + " [" + str(np.round(np.min(ca), 2)) + "-" + str(np.round(np.max(ca), 2)) +"]")
    statistic = stats.shapiro(cb)
    file.write("\nOriginal Shapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))
    statistic = stats.shapiro(ca)
    file.write("\nStandardized Shapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))
    statistic = stats.ttest_rel(cb, ca)
    file.write("\nOriginal vs. Standardized ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.wilcoxon(cb, ca)
    file.write("\nOriginal vs. Standardized Wilcoxon p = " + str(np.round(statistic.pvalue, 3)))

    file.write("\n\nAmyloidosis - number of maps: " + str(cohort_size[2]))
    cb = covs_before[cohort_size[0]+cohort_size[1]:]
    ca = covs_after[cohort_size[0]+cohort_size[1]:]
    file.write("\nOriginal: " + str(np.round(np.mean(cb), 2)) + "+/-" + str(np.round(np.std(cb), 2)) + " [" + str(np.round(np.min(cb), 2)) + "-" + str(np.round(np.max(cb), 2)) +"]")
    file.write("\nStandardized: " + str(np.round(np.mean(ca), 2)) + "+/-" + str(np.round(np.std(ca), 2)) + " [" + str(np.round(np.min(ca), 2)) + "-" + str(np.round(np.max(ca), 2)) +"]")
    statistic = stats.shapiro(cb)
    file.write("\nOriginal Shapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))
    statistic = stats.shapiro(ca)
    file.write("\nStandardized Shapiro-Wilk-Test p:" + str(np.round(statistic[-1],3)))
    statistic = stats.ttest_rel(cb, ca)
    file.write("\nOriginal vs. Standardized ttest p = " + str(np.round(statistic.pvalue, 3)))
    statistic = stats.wilcoxon(cb, ca)
    file.write("\nOriginal vs. Standardized Wilcoxon p = " + str(np.round(statistic.pvalue, 3)))

    file.close()
