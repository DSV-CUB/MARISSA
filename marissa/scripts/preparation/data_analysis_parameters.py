import os
import pydicom
import numpy as np

path_data = r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\KRANKE\KRANKE_FINAL"
path_out = r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\KRANKE\OVERVIEW"

#path_data = r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\GESUNDE_FINAL"
#path_out = r"C:\Users\CMRT\Documents\DSV\Dicom Daten\Mapping DATA\GESUNDE\OVERVIEW"


parameters = []
parameters.append(["age", "int(str(dcm[0x00101010].value)[:-1])"]) # age
parameters.append(["gender", "dcm[0x00100040].value"]) # sex
parameters.append(["height", "dcm[0x00101020].value"]) # height/size
parameters.append(["weight", "dcm[0x00101030].value"]) # weight
parameters.append(["sequence", "(\"ShMOLLI\" if \"shmolli\" in str(dcm[0x0008, 0x103e].value).lower() else (\"SASHA\" if \"sasha\" in str(dcm[0x0008, 0x103e].value).lower() else \"MOLLI\"))"]) # sequence
parameters.append(["device", "str(dcm[524400].value) + \"#\" + str(dcm[528528].value) + \"#\" + str(dcm[1576992].value)"]) # device
parameters.append(["BMI group 3", "(-1 if ((float(dcm[0x0010, 0x1030].value) if float(dcm[0x0010, 0x1030].value) < 1000 else float(dcm[0x0010, 0x1030].value) / 1000) / (float(dcm[0x0010, 0x1020].value) if float(dcm[0x0010, 0x1020].value) < 2.5 else float(dcm[0x0010, 0x1020].value) / 100) ** 2) < 18.5 else (0 if ((float(dcm[0x0010, 0x1030].value) if float(dcm[0x0010, 0x1030].value) < 1000 else float(dcm[0x0010, 0x1030].value) / 1000) / (float(dcm[0x0010, 0x1020].value) if float(dcm[0x0010, 0x1020].value) < 2.5 else float(dcm[0x0010, 0x1020].value) / 100) ** 2) < 25 else 1))"]) # BMI group



studies = os.listdir(path_data)
for study in studies:
    info = []
    try:
        cases = os.listdir(os.path.join(path_data, study))
    except:
        continue

    for case in cases:

        for root, _, files in os.walk(os.path.join(path_data, study, case)):
            for file in files:
                if file.endswith(".dcm"):
                    dcm = pydicom.dcmread(os.path.join(root, file), force=True)

                    case_info = []

                    for parameter in parameters:
                        try:
                            case_info.append(str(eval(parameter[1])))
                        except:
                            case_info.append("")

                    case_info = [study, case, file] + case_info

                    info.append(case_info)

    info = np.array(info, dtype=str)

    text = study + " has " + str(len(cases)) + " cases with " + str(len(info)) + " measurements\n\n"
    offset = 3
    for i in range(len(parameters)):
        value, count = np.unique(info[:,i+offset], return_counts=True)

        text = text + parameters[i][0] + "\n"
        text = text + "\n".join([value[j] + "\t" + str(count[j]) for j in range(len(value))]) + "\n\n"

    file = open(os.path.join(path_out, study + ".txt"), "a")
    file.writelines(text)
    file.close()

info = []
for root, _, files in os.walk(path_data):
    for file in files:
        if file.endswith(".dcm"):
            dcm = pydicom.dcmread(os.path.join(root, file), force=True)

            case_info = []

            for parameter in parameters:
                try:
                    case_info.append(str(eval(parameter[1])))
                except:
                    case_info.append("")

            case_info = [os.path.join(root, file), str(dcm[0x0008, 0x0018].value)] + case_info

            info.append(case_info)

#info = np.array(info, dtype=str)

info = "\n".join(["\t".join(info[ii]) for ii in range(len(info))])

info = "File\tSUID\t" + "\t".join(parameters[ii][0] for ii in range(len(parameters))) + "\n" + info

file = open(os.path.join(path_out, "#OVERVIEW.txt"), "a")
file.writelines(info)
file.close()

