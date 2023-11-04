import numpy as np


def get_BPSP_setupID(project, filepath):
    file = open(filepath, "r")
    readdata = file.readlines()
    file.close()
    setup_info = [readdata[ii].split("\t") for ii in range(1, len(readdata))]
    evalS2 = np.array(setup_info)[:,-1].flatten().astype(float)
    index_top = np.argsort(evalS2)[0] # lowest CoV is best
    best_setup = setup_info[index_top]
    setupID = project.select("SELECT setupID FROM tbl_setup WHERE description = '" + best_setup[0] + "'")[0][0]
    return setupID


def get_pids(project, parameters):
    pids = []
    for pd in parameters:
        pids.append(project.select("SELECT parameterID FROM tbl_parameter WHERE description = '" + pd + "'")[0][0])
    return pids


def get_data_info(project, description, pids):
    selection = project.select("SELECT s.SOPinstanceUID, s.segmentationID FROM (tbl_segmentation AS s INNER JOIN tbl_data AS d ON s.SOPinstanceUID = d.SOPinstanceUID) WHERE d.description='" + description + "'")
    data = []
    info_data = []
    for i in range(len(selection)):
        data.append([selection[i][0], selection[i][1]])
        info_data.append(project.get_data_parameters(selection[i][0], pids))
    info_data = np.array(info_data)
    return data, info_data

def get_all_y(project, data, setupID):
    all_y = []

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
        all_y.append(prog_y)

    return all_y