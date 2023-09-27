import pydicom
import os
import numpy as np
from marissa.toolbox.tools import tool_pydicom, tool_general

#paths_cohorts = [r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL", r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\KRANKE\KRANKE_FINAL\HCM", r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\KRANKE\KRANKE_FINAL\LVHSTRAIN_Amyloidose", r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL_TRAINING", r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL_TEST"]
#cohort_description = ["Healthy", "HCM", "Amyloidosis", "Healthy Training", "Healthy Test"]

paths_cohorts = [r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\SCMR\GESUNDE_FINAL", r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\SCMR\SCMR_GESUNDE_FINAL_TEST", r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\SCMR\SCMR_GESUNDE_FINAL_TRAINING"]
cohort_description = ["Healthy", "Healthy Test", "Healthy Training"]


for i in range(len(paths_cohorts)):

    counter_M = 0
    counter_N = 0
    studyIDs = []
    age = []
    sex = []
    system = []
    sequence = []

    for root, _, files in os.walk(paths_cohorts[i]):
        for file in files:
            try:
                dcm = pydicom.dcmread(os.path.join(root, file), force=True)
                STUID = str(dcm[0x0020, 0x000D].value)

                counter_M = counter_M + 1

                if STUID in studyIDs or ("DZHK TV" in root and not "Helios" in root) or ("TravellingVolunteers" in root and not "DHZB" in root) or ("zScore" in root and not "Berlin Probanden 4" in root):
                    # same subject were measured on different systems, account to not count them twice
                    pass
                else:
                    studyIDs.append(STUID)
                    counter_N = counter_N + 1
                    age.append(tool_pydicom.get_value(dcm[0x0010, 0x1010].value, "AS"))
                    sex.append(str(dcm[0x0010, 0x0040].value))

                system.append(str(dcm[0x0018, 0x0087].value) + "T " + str(dcm[0x0008, 0x0070].value) + " " + str(dcm[0x0008, 0x1090].value) + " [" + str(dcm[0x0018, 0x1020].value) + "]")
                sequence.append(["SASHA GRE" if ("SASHAGRE" in sd) else "SASHA" if ("SASHA" in sd) else "ShMOLLI" if ("SHMOLLI" in sd) else "MOLLI 3(3)5 s" if ("MOLLI335S" in sd) else "MOLLI 3(3)5 b" if ("MOLLI335" in sd) else "MOLLI 3(3)3(3)5 s" if ("MOLLI33335S" in sd) else "MOLLI 3(3)3(3)5 b" if ("MOLLI33335" in sd) else "MOLLI 4(1)3(1)2 s" if ("MOLLI41312S" in sd) else "MOLLI 4(1)3(1)2 b" if ("MOLLI41312" in sd) else "MOLLI 5(3)3 s" if ("MOLLI533S" in sd or "MOLLI5S3S3S" in sd) else "MOLLI 5(3)3 b" for sd in [tool_general.string_stripper(str(dcm[0x0008, 0x103e].value).upper().replace('SAX',''), [])]][0])

                #print(tool_general.string_stripper(str(dcm[0x0008, 0x103e].value).upper(),[]) + "\t" + sequence[-1])
            except:
                pass

    print(cohort_description[i])
    print("N = " + str(counter_N))
    print("M = " + str(counter_M))
    print("age = " + str(np.round(np.mean(age), 2)) + " +/- " + str(np.round(np.std(age), 2)))
    u, c = np.unique(sex, return_counts=True)
    print("sex = " + " | ".join([str(c[ii]) + " " + str(u[ii]) for ii in range(len(c))]))
    print("systems")
    u, c = np.unique(system, return_counts=True)
    for ii in range(len(c)):
        print(str(c[ii]) + " " + str(u[ii]))
    print("sequences")
    u, c = np.unique(sequence, return_counts=True)
    for ii in range(len(c)):
        print(str(c[ii]) + " " + str(u[ii]))
    print("\n\n")