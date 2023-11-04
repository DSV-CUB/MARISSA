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
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL"
#parameters = ["PatientsAge", "PatientsSex", "BMIgroup3", "System", "T1Sequence"]
#reference = ["18Y", "M", 0, "SIEMENS#Verio#syngo MR B17", "MOLLI"]

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 533"]
reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 533"]

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

file_s2 = os.path.join(path_out, "train_step2_top3_binning.txt")
file = open(file_s2, "r")
readdata = file.readlines()
file.close()
setup_info2 = [readdata[ii].split("\t") for ii in range(1, len(readdata))]
evalS2 = np.array(setup_info2)[:,-1].flatten().astype(float)
index_top = np.argsort(evalS2)[0] # lowest CoV is best
best_setup = setup_info2[index_top]
setupID = project.select("SELECT setupID FROM tbl_setup WHERE description = '" + best_setup[0] + "'")[0][0]

########################################################################################################################
# STATISTICS ###########################################################################################################
########################################################################################################################
file_info = os.path.join(path_out, "intra_cohort_analysis.txt")
if os.path.exists(file_info):
    os.remove(file_info)

file = open(file_info, "a")

counter = 0
for d in [data_test, data_patient_hcm, data_patient_amy]:

    t1mean_before = []
    t1mean_after = []
    sys_seq = []

    for i in range(len(d)):
        dcm = project.get_data(d[i][0])[0]
        mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(d[i][1]))[0][0]
        try:
            md = project.get_standardization(dcm, mask, setupID, False)
        except:
            continue
        t1mean_before.append(np.mean(md.value_progression[0]))
        t1mean_after.append(np.mean(md.value_progression[-1]))
        if counter == 0:
            sys_seq.append(info_data_test[i,2] + " # T1 Map " + info_data_test[i,3])
        elif counter == 1:
            sys_seq.append(info_data_patient_hcm[i,2] + " # T1 Map " + info_data_patient_hcm[i,3])
        elif counter == 2:
            sys_seq.append(info_data_patient_amy[i,2] + " # T1 Map " + info_data_patient_amy[i,3])

    if counter == 0:
        file.write("Healthy")
    elif counter == 1:
        file.write("\n\nHCM")
    elif counter == 2:
        file.write("\n\nAmyloidosis")

    file.write("\nOriginal")
    for ssu in np.unique(sys_seq):
        indeces = np.argwhere(np.array(sys_seq) == ssu).flatten()
        cov = 100 * np.std(np.array(t1mean_before)[indeces]) / np.mean(np.array(t1mean_before)[indeces])
        file.write("\n" + ssu + " CoV[%] = " + str(np.round(cov, 2)))
    cov = 100 * np.std(t1mean_after) / np.mean(t1mean_after)
    file.write("\nStandardized\n" + ssu + " CoV[%] = " + str(np.round(cov, 2)))

    counter = counter + 1
file.close()