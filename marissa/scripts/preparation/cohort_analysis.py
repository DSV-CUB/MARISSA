from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics
from marissa.scripts.plots import basic_functions
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
import warnings
import pydicom
warnings.filterwarnings('ignore')

########################################################################################################################
# Match subjects to DICOMs #############################################################################################
########################################################################################################################

paths = []
paths.append(r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL_Rearrange_Subjects")
paths.append(r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\KRANKE\KRANKE_FINAL\HCM")
paths.append(r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\KRANKE\KRANKE_FINAL\LVHSTRAIN_Amyloidose")

subjectmatrix = []
for cohort_path in paths:
    studies = os.listdir(cohort_path)
    for study in studies:
        if study.startswith("#"):
            continue
        subjects = os.listdir(os.path.join(cohort_path, study))
        for subject in subjects:
            pathindiv = os.path.join(cohort_path,study,subject)

            for root, _, files in os.walk(pathindiv):
                for file in files:
                    if file.endswith(".dcm"):
                        dcm = pydicom.dcmread(os.path.join(root, file))
                        SUID = str(dcm[0x0008, 0x0018].value)
                        subjectmatrix.append([SUID, study + "#" + subject])
subjectmatrix = np.array(subjectmatrix)

########################################################################################################################
# USER INPUT ###########################################################################################################
########################################################################################################################

project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\DoktorarbeitDSV.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER"
path_training = os.path.join(path_out, "train_step2_top3_binning.txt")

parameters = ["SOPInstanceUID", "PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["SUID", "age", "sex", "system", "sequence"]
#reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]
#reference_str = ["18.0", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]

colours = ["#008000", "#ff8000", "#ff0000"]
cohorts = ["Healthy", "HCM", "Amyloidosis"]

########################################################################################################################
# PREPARATION ##########################################################################################################
########################################################################################################################

project = marissadb.Module(project_path)
setupID = basic_functions.get_BPSP_setupID(project, path_training)
pids = basic_functions.get_pids(project, parameters)

data_healthy, info_data_healthy = basic_functions.get_data_info(project, ["TRAINING", "TESTHEALTHY"], pids)
data_train, info_data_train = basic_functions.get_data_info(project, "TRAINING", pids)
data_test, info_data_test = basic_functions.get_data_info(project, "TESTHEALTHY", pids)
data_patient_hcm, info_data_patient_hcm = basic_functions.get_data_info(project, "TESTPATIENTHCM", pids)
data_patient_amy, info_data_patient_amy = basic_functions.get_data_info(project, "TESTPATIENTAMYLOIDOSE", pids)

all_y_healthy = basic_functions.get_all_y(project, data_healthy, setupID)
all_y_healthy_test = basic_functions.get_all_y(project, data_test, setupID)
all_y_healthy_train = basic_functions.get_all_y(project, data_train, setupID)
all_y_patient_hcm = basic_functions.get_all_y(project, data_patient_hcm, setupID)
all_y_patient_amy = basic_functions.get_all_y(project, data_patient_amy, setupID)

healthy_before = np.array(all_y_healthy)[:,0].flatten()
healthy_after = np.array(all_y_healthy)[:,-1].flatten()
healthy_test_before = np.array(all_y_healthy_test)[:,0].flatten()
healthy_test_after = np.array(all_y_healthy_test)[:,-1].flatten()
healthy_train_before = np.array(all_y_healthy_train)[:,0].flatten()
healthy_train_after = np.array(all_y_healthy_train)[:,-1].flatten()
patient_hcm_before = np.array(all_y_patient_hcm)[:,0].flatten()
patient_hcm_after = np.array(all_y_patient_hcm)[:,-1].flatten()
patient_amy_before = np.array(all_y_patient_amy)[:,0].flatten()
patient_amy_after = np.array(all_y_patient_amy)[:,-1].flatten()

########################################################################################################################
# ANALYSIS #############################################################################################################
########################################################################################################################

###
###
### incorporate subject information here !!!!
###
###
def print_cohort_variations(cohort_info, cohort_y, all_info, subjectmatch):
    for i in range(len(all_info)):
        indeces = np.argwhere(np.all(cohort_info[:,3:]==all_info[i,:], axis=1)).flatten()
        if len(indeces) > 0:
            subject_done = []
            N = 0
            Nm = 0
            Nf = 0
            M = 0
            Mm = 0
            Mf = 0
            age = []
            age_m = []
            age_f = []
            t1 = []
            t1_m = []
            t1_f = []
            for j in range(len(indeces)):
                SUID = cohort_info[indeces[j], 0]
                subject = subjectmatch[np.argwhere(subjectmatch[:,0].flatten()==SUID), 1].flatten()[0]
                M = M + 1
                t1.append(cohort_y[indeces[j]])
                if cohort_info[indeces[j], 2] == "M":
                    Mm = Mm + 1
                    t1_m.append(cohort_y[indeces[j]])
                elif cohort_info[indeces[j], 2] == "F":
                    Mf = Mf + 1
                    t1_f.append(cohort_y[indeces[j]])

                if subject in subject_done:
                    continue
                else:
                    subject_done.append(subject)
                    N = N +1
                    age.append(float(cohort_info[indeces[j],1]))
                    if cohort_info[indeces[j], 2] == "M":
                        Nm = Nm + 1
                        age_m.append(float(cohort_info[indeces[j],1]))
                    elif cohort_info[indeces[j], 2] == "F":
                        Nf = Nf + 1
                        age_f.append(float(cohort_info[indeces[j],1]))


            print(all_info[i,0] + " " + all_info[i,1] + "\nN=" + str(N) + " | M=" + str(M) + "\nNm=" + str(Nm) + " | Mm=" + str(Mm) + "\nT1m=" + '{:.2f}'.format(np.mean(t1_m)) + "\u00b1" + '{:.2f}'.format(np.std(t1_m)) + "\nam=" + '{:.2f}'.format(np.mean(age_m)) + "\u00b1" + '{:.2f}'.format(np.std(age_m)) + "\nNf=" + str(Nf) + " | Mf=" + str(Mf) +"\nT1f=" + '{:.2f}'.format(np.mean(t1_f)) + "\u00b1" + '{:.2f}'.format(np.std(t1_f)) + "\naf=" + '{:.2f}'.format(np.mean(age_f)) + "\u00b1" + '{:.2f}'.format(np.std(age_f)))
        else:
            print(all_info[i,0] + " " + all_info[i,1] + " X")
    return

scaseqvar = np.unique(np.append(np.append(np.append(info_data_train[:, 3:], info_data_test[:, 3:],axis=0), info_data_patient_hcm[:, 3:], axis=0), info_data_patient_amy[:, 3:], axis=0), axis=0)

print("################################# HEALTHY ##################################\n")
print_cohort_variations(info_data_healthy, healthy_before, scaseqvar, subjectmatrix)
print("\n\n ALL T1 = " + '{:.2f}'.format(np.mean(healthy_before)) + "\u00b1" + '{:.2f}'.format(np.std(healthy_before)) + "\n\n")

print("\n################################# HEALTHY TRAIN ##################################\n")
print_cohort_variations(info_data_train, healthy_train_before, scaseqvar, subjectmatrix)
print("\n\n ALL T1 = " + '{:.2f}'.format(np.mean(healthy_train_before)) + "\u00b1" + '{:.2f}'.format(np.std(healthy_train_before)) + "\n\n")

print("\n################################# HEALTHY TEST ##################################\n")
print_cohort_variations(info_data_test, healthy_test_before, scaseqvar, subjectmatrix)
print("\n\n ALL T1 = " + '{:.2f}'.format(np.mean(healthy_test_before)) + "\u00b1" + '{:.2f}'.format(np.std(healthy_test_before)) + "\n\n")

print("\n################################# HCM ##################################\n")
print_cohort_variations(info_data_patient_hcm, patient_hcm_before, scaseqvar, subjectmatrix)
print("\n\n ALL T1 = " + '{:.2f}'.format(np.mean(patient_hcm_before)) + "\u00b1" + '{:.2f}'.format(np.std(patient_hcm_before)) + "\n\n")

print("\n################################# AMY ##################################\n")
print_cohort_variations(info_data_patient_amy, patient_amy_before, scaseqvar, subjectmatrix)
print("\n\n ALL T1 = " + '{:.2f}'.format(np.mean(patient_amy_before)) + "\u00b1" + '{:.2f}'.format(np.std(patient_amy_before)) + "\n\n")

