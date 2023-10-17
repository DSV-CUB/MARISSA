from pyclustering.container.cftree import leaf_node

from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy import stats
import pydicom
import warnings
warnings.filterwarnings('ignore')

# SETUP
project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\DoktorarbeitDSV.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER"
paths_studies = [r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL_TEST_Re-Arrange_intra-subject\DZHK TV", r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL_TEST_Re-Arrange_intra-subject\TravellingVolunteers", r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL_TEST_Re-Arrange_intra-subject\zScore"]

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]
reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]

# PREPARATION
project = marissadb.Module(project_path)

pids = []
for pd in parameters:
    pids.append(project.select("SELECT parameterID FROM tbl_parameter WHERE description = '" + pd + "'")[0][0])

# Healthy TEST
selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TESTHEALTHY'")
data_test = []
info_data_test = []
for i in range(len(selection)):
    data_test.append([selection[i][0], selection[i][1]])
    info_data_test.append(project.get_data_parameters(selection[i][0], pids))
info_data_test = np.array(info_data_test)

# Healthy Training
selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TRAINING'")
data = []
info_data = []
for i in range(len(selection)):
    data.append([selection[i][0], selection[i][1]])
    info_data.append(project.get_data_parameters(selection[i][0], pids))
info_data = np.array(info_data)

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

parameter_healthy = [" | ".join(info_data_test[:, 2:].tolist()[ii]) for ii in range(len(info_data_test))] + [" | ".join(info_data_test[:, 2:].tolist()[ii]) for ii in range(len(info_data_test))]
parameter_healthy_train = [" | ".join(info_data[:, 2:].tolist()[ii]) for ii in range(len(info_data))] + [" | ".join(info_data[:, 2:].tolist()[ii]) for ii in range(len(info_data))]
parameter_HCM = [" | ".join(info_data_patient_hcm[:, 2:].tolist()[ii]) for ii in range(len(info_data_patient_hcm))]
parameter_AMY = [" | ".join(info_data_patient_amy[:, 2:].tolist()[ii]) for ii in range(len(info_data_patient_amy))]
unique_settings = np.unique(parameter_healthy + parameter_healthy_train + parameter_HCM + parameter_AMY)




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
# Value Spread #########################################################################################################
########################################################################################################################
# CI
selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='TRAINING'")
data = []
info_data = []
for i in range(len(selection)):
    data.append([selection[i][0], selection[i][1]])
    info_data.append(project.get_data_parameters(selection[i][0], pids))
info_data = np.array(info_data)

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

miny = np.inf
maxy = 0

cases_info = []
for study in paths_studies:
    cases = os.listdir(study)
    for case in cases:
        path_run = os.path.join(study, case)
        case_info = []
        for root, _, files in os.walk(path_run):
            for file in files:
                try:
                    dcm = pydicom.dcmread(os.path.join(root, file))
                    SUID = str(dcm[0x0008, 0x0018].value)
                    sd = str(dcm[0x0008, 0x103e].value)

                    #if "sasha" in sd.lower():
                    #    continue
                    case_info.append(SUID)
                except:
                    pass
        case_info = "('" + "','".join(case_info) + "')"

        selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE s.SOPinstanceUID IN " + case_info)

        case_info = {}
        case_info["SOPinstanceUID"] = []
        case_info["parameters"] = []
        case_info["value_progression"] = []
        for i in range(len(selection)):
            dcm = project.get_data(selection[i][0])[0]
            mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(selection[i][1]))[0][0]
            try:
                md = project.get_standardization(dcm, mask, setupID, False)
            except:
                continue

            case_info["SOPinstanceUID"].append(str(dcm[0x0008, 0x0018].value))
            case_info["value_progression"].append(np.mean(md.value_progression, axis=1))
            case_info["parameters"].append(project.get_data_parameters(selection[i][0], pids))

            miny = np.min([miny] + np.mean(md.value_progression, axis=1).tolist())
            maxy = np.max([maxy] + np.mean(md.value_progression, axis=1).tolist())

        cases_info.append(case_info)

sys_seq = []
for i in range(len(cases_info)):
    for j in range(len(cases_info[i]["parameters"])):
        #sys_seq.append(cases_info[i]["parameters"][j][2].replace("#", " | ") + " | T1 Map " + cases_info[i]["parameters"][j][3])
        sys_seq.append(cases_info[i]["parameters"][j][2] + " | " + cases_info[i]["parameters"][j][3])
sys_seq = np.unique(sys_seq)


cmap = cm.get_cmap('nipy_spectral')

#fig, ax = plt.subplots(int(np.ceil(len(cases_info)/3)), 4, figsize=(30, int(np.ceil(len(cases_info)/2)) * 5))
fig, ax = plt.subplots(5, 2, figsize=(20, 35))

counter = 0
for j in range(4):
    for i in range(2):
        try:
            case = cases_info[counter]
        except:
            continue
        original_min = np.inf
        original_max = 0
        original_values = []
        standardized_min = np.inf
        standardized_max = 0
        standardized_values = []

        for k in range(len(case["value_progression"])):
            y = case["value_progression"][k]
            sys_seq_info = case["parameters"][k][2] + " | " + case["parameters"][k][3]
            index = np.argwhere(unique_settings==sys_seq_info).flatten()[0]
            rgba = cmap(index/(len(unique_settings)))

            original_min = np.min([original_min, y[0]])
            original_max = np.max([original_max, y[0]])
            original_values.append(y[0])
            standardized_min = np.min([standardized_min, y[-1]])
            standardized_max = np.max([standardized_max, y[-1]])
            standardized_values.append(y[-1])

            ax[j, i].scatter([0], (y[0]), c="#660066", zorder=30)
            ax[j, i].scatter([1], (y[-1]), c="#008800", zorder=30)

            ax[j, i].plot([0, 1], [y[0], y[-1]], c=rgba, ls="--", lw=2, zorder=20)

        r1 = plt.Rectangle((-0.5, original_min), 0.5, original_max-original_min, fill=True, color="#66006644", zorder=10)
        ax[j, i].add_patch(r1)
        #ax[j, i].text(-0.25, ((original_max-original_min) / 2 + original_min), "\u0394T1\n=\n" + "{:.2f}".format(np.round(original_max-original_min, 2)) + " ms", horizontalalignment='center', verticalalignment='center', color="#660066", fontsize=11, rotation=0)
        omin = np.min(original_values)
        omax = np.max(original_values)
        ocov = 100 * np.std(original_values) / np.mean(original_values)
        ax[j, i].text(-0.25, ((omax-omin) / 2 + omin), "\u0394T1 =\n" + "{:.2f}".format(np.round(omax-omin, 2)) + " ms\n\nCOV =\n" + "{:.2f}".format(np.round(ocov, 2)) + " %", horizontalalignment='center', verticalalignment='center', color="#660066", fontsize=20, rotation=0)

        r1 = plt.Rectangle((1, standardized_min), 0.5, standardized_max-standardized_min, fill=True, color="#00880044", zorder=10)
        ax[j, i].add_patch(r1)
        #ax[j, i].text(len(y)-0.75, ((standardized_max-standardized_min) / 2 + standardized_min), "\u0394T1\n=\n" + "{:.2f}".format(np.round(standardized_max-standardized_min, 2)) + " ms", horizontalalignment='center', verticalalignment='center', color="#008800", fontsize=11, rotation=0)
        smin = np.min(standardized_values)
        smax = np.max(standardized_values)
        scov = 100 * np.std(standardized_values) / np.mean(standardized_values)
        ax[j, i].text(1.25, ((smax-smin) / 2 + smin), "\u0394T1 =\n" + "{:.2f}".format(np.round(smax-smin, 2)) + " ms\n\nCOV =\n" + "{:.2f}".format(np.round(scov, 2)) + " %", horizontalalignment='center', verticalalignment='center', color="#008800", fontsize=20, rotation=0)

        ax[j, i].set_xlim((-0.5, 1.5))
        ax[j, i].set_ylim((miny-10, maxy+10))
        ax[j, i].set_ylabel("mean T1 [ms]", fontsize=20)
        ax[j, i].set_xticks([0, 1], ["original\nvalues", "standardized\nvalues"], fontsize=20)

        #ax[j, i].grid(lw=1, c="#bbbbbb")
        for t in ax[j, i].get_yticks():
            ax[j, i].plot([0, 1], [t, t], c="#bbbbbb", zorder=1)
        ax[j, i].axvline(0, c="#bbbbbb", zorder=1)
        ax[j, i].axvline(1, c="#bbbbbb", zorder=1)

        counter2 = 0
        for tl in ax[j, i].get_xticklabels():
            if counter2 == 0:
                tl.set_color("#660066")
            else:
                tl.set_color("#008800")
            counter2 = counter2+1

        #if len(y) > 2:
        ax[j, i].set_xticks([0.5], labels=["\n\u2192 BPSP \u2192"], minor=True, fontsize=20, c="#444444")

        ax[j,i].yaxis.set_tick_params(labelsize=20)

        #ax[j, i].legend(loc="upper right", fontsize=9)
        counter = counter + 1

#ax[-1, -2].axis("off")

gs = ax[-1, 0].get_gridspec()
# remove the underlying axes
for axr in ax[-1, 0:]:
    axr.remove()
axbig = fig.add_subplot(gs[-1, 0:])

legend_elements = []
for i in range(len(unique_settings)):
    isin = np.argwhere(sys_seq==unique_settings[i])

    if len(isin) > 0:
        rgba = cmap(i/(len(unique_settings)))
        legend_elements.append(Line2D([0], [0], color=rgba, lw=4, ls="--", label=unique_settings[i].replace("#", " | ").replace(" Healthcare ", " ").replace(" Medical Systems ", " ").replace("_fit", "Fit")))


#ax[-1, -1].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=20)
#ax[-1, -1].axis("off")
axbig.axis("off")
legend = axbig.legend(handles=legend_elements, loc='center', frameon=True, fontsize=20, ncol=2)
legend.get_frame().set_facecolor('#efefef66')

plt.tight_layout()
plt.savefig(os.path.join(path_out, "intra_subject_value_progression.jpg"), dpi=600)
a = 0