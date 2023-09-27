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
project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\SCMR_Abstract.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_SCMR"
#parameters = ["PatientsAge", "PatientsSex", "BMIgroup3", "System", "T1Sequence"]
#reference = ["18Y", "M", 0, "SIEMENS#Verio#syngo MR B17", "MOLLI"]

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "3.0T SIEMENS Verio [syngo MR B17]", "MOLLI 5(3)3 b"]

run_step1 = True
run_step2 = False

skip_s1 = -1
skip_s2 = -1

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

# STEP 1: no binning
file_s1 = os.path.join(path_out, "train_step1_no_binning.txt")
if run_step1 or not os.path.exists(file_s1):
    print("### Run Step 1: No Binning ###")
    bins = 1
    clustertype = "kmeans"
    counter = -1
    setup_info1 = []

    if os.path.exists(file_s1):
        os.remove(file_s1)

    with open(file_s1, "a") as file:
        file.write("description\tbins\tclustertype\tregressiontype\tytype\tmode\tCoV %\n")
        file.close()


    for regressiontype in ["extratrees", "linear", "randomforest", "linearsvr"]:
        for ytype in ["relative", "absolute"]:
            for mode in ["individual", "cascaded", "ensemble"]:
                counter = counter + 1
                if counter <= skip_s1:
                    continue

                description = "S1T" + str(counter).zfill(2)


                # fill tbl_setup and tbl_match_setup_parameter
                try:
                    project.delete_setup(project.select("SELECT setupID FROM tbl_setup WHERE description='" + description + "'")[0][0])
                except:
                    pass
                setupID = project.insert_setup(description, bins, clustertype, regressiontype, ytype, mode, parameterIDs=pids)

                # fill tbl_match_setup_data_segmentation
                for i in range(len(data)):
                    project.execute("INSERT OR IGNORE INTO tbl_match_setup_data_segmentation VALUES (" + str(setupID) + ", '" + data[i][0] + "', " + str(data[i][1]) + ")")

                # run standardization training
                #try:
                project.insert_standardization(setupID, reference=reference)
                #except:
                #    print("FAIL TRAINING OF " + description)
                #    continue


                # run testing
                # get_standardization(self, dcm, mask, setupID, skip_unknown=True)
                meant1 = []
                for i in range(len(data_test)):
                    dcm = project.get_data(data_test[i][0])[0]
                    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data_test[i][1]))[0][0]
                    md = project.get_standardization(dcm, mask, setupID, True)
                    meant1.append(np.mean(md.value_progression[-1]))
                CoV = 100 * (np.std(meant1)/np.mean(meant1))

                setup_info1.append([description, bins, clustertype, regressiontype, ytype, mode, CoV])
                with open(file_s1, "a") as file:
                    file.write(description + "\t" + str(bins) + "\t" + clustertype + "\t" + regressiontype + "\t" + ytype + "\t" + mode + "\t" + str(CoV) + "\n")
                    file.close()

                print(description + "\t" + str(bins) + "\t" + clustertype + "\t" + regressiontype + "\t" + ytype + "\t" + mode + "\t" + str(CoV))
    print("### End Step 1 ###")


file = open(file_s1, "r")
readdata = file.readlines()
file.close()
setup_info1 = [readdata[ii].split("\t") for ii in range(1, len(readdata))]


# STEP 2: Top 3 from STEP 1 with binning b = 2 ... 20
file_s2 = os.path.join(path_out, "train_step2_top3_binning.txt")

if run_step2 or not os.path.exists(file_s2):
    print("### Run Step 2: Binning of Top 3 ###")
    evalS1 = np.array(setup_info1)[:,-1].flatten().astype(float)
    setup_rym = np.array(setup_info1, dtype=object)[:,3:6]

    indeces_top3 = np.argsort(evalS1)[:3] # lowest CoV is best
    setup_rym_top3 = setup_rym[indeces_top3,:][::-1,:]
    setup_rym_top3

    if os.path.exists(file_s2):
        os.remove(file_s2)

    with open(file_s2, "a") as file:
        file.write("description\tbins\tclustertype\tregressiontype\tytype\tmode\tCoV %\n")
        file.close()

    for i in range(len(setup_info1)):
        with open(file_s2, "a") as file:
            file.write(str(setup_info1[i][0]) + "\t" + str(setup_info1[i][1]) + "\t" + str(setup_info1[i][2]) + "\t" + str(setup_info1[i][3]) + "\t" + str(setup_info1[i][4]) + "\t" + str(setup_info1[i][5]) + "\t" + str(setup_info1[i][6]))
            file.close()

    counter = -1
    setup_info2 = copy.deepcopy(setup_info1)

    for srym in setup_rym_top3:
        regressiontype = srym[0]
        ytype = srym[1]
        mode = srym[2]
        for clustertype in ["aglomerative_average", "aglomerative_complete", "aglomerative_single", "aglomerative_ward", "equaldistant", "equalsize", "gaussian_mixture", "kmeans"]: #, "spectral"]:
            for bins in range(2, 11, 1):

                counter = counter + 1
                if counter <= skip_s2:
                    continue

                description = "S2T" + str(counter).zfill(2)

                # fill tbl_setup and tbl_match_setup_parameter
                try:
                    project.delete_setup(project.select("SELECT setupID FROM tbl_setup WHERE description='" + description + "'")[0][0])
                except:
                    pass

                setupID = project.insert_setup(description, bins, clustertype, regressiontype, ytype, mode, parameterIDs=pids)

                # fill tbl_match_setup_data_segmentation
                for i in range(len(data)):
                    project.execute("INSERT OR IGNORE INTO tbl_match_setup_data_segmentation VALUES (" + str(setupID) + ", '" + data[i][0] + "', " + str(data[i][1]) + ")")

                # run standardization training
                try:
                    project.insert_standardization(setupID, reference=reference)
                except:
                    print("FAIL TRAINING " + description)
                    continue

                # run testing
                # get_standardization(self, dcm, mask, setupID, skip_unknown=True)
                meant1 = []
                for i in range(len(data_test)):
                    dcm = project.get_data(data_test[i][0])[0]
                    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data_test[i][1]))[0][0]
                    md = project.get_standardization(dcm, mask, setupID, True)
                    meant1.append(np.mean(md.value_progression[-1]))
                CoV = 100 * (np.std(meant1)/np.mean(meant1))

                setup_info2.append([description, bins, clustertype, regressiontype, ytype, mode, CoV])
                with open(file_s2, "a") as file:
                    file.write(description + "\t" + str(bins) + "\t" + clustertype + "\t" + regressiontype + "\t" + ytype + "\t" + mode + "\t" + str(np.round(CoV, 2)) + "\n")
                    file.close()

                print(description + "\t" + str(bins) + "\t" + clustertype + "\t" + regressiontype + "\t" + ytype + "\t" + mode + "\t" + str(np.round(CoV, 2)))
    print("### End Step 2 ###")
else:
    file = open(file_s2, "r")
    readdata = file.readlines()
    file.close()
    setup_info2 = [readdata[ii].split("\t") for ii in range(1, len(readdata))]