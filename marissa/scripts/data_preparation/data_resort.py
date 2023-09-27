import numpy as np
import os
import shutil
from marissa.modules.database import marissadb
from marissa.modules.pesf import pesf
from marissa.toolbox.tools import tool_general, tool_R

if False:
    path_db = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\DB"
    #path_data = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw"
    path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw_sorted_NoAnonym"
    mdb = marissadb.Module(path=path_db)

    selection = mdb.select("SELECT UID FROM tbl_data WHERE "
                           "(NOT tag524400 IS NULL AND tag524400 != 'None' AND tag524400 != '') AND "
                           "(NOT tag528446 IS NULL AND tag528446 != 'None' AND tag528446 != '') AND "
                           "(NOT tag528528 IS NULL AND tag528528 != 'None' AND tag528528 != '') AND "
                           "(NOT tag1048640 IS NULL AND tag1048640 != 'None' AND tag1048640 != '') AND "
                           "(NOT tag1052688 IS NULL AND tag1052688 != 'None' AND tag1052688 != '' AND CAST(SUBSTR(tag1052688,1,3) AS INT) < 90) AND "
                           "(NOT tag1052704 IS NULL AND tag1052704 != 'None' AND tag1052704 != '' AND CAST(tag1052704 AS REAL) > 1) AND "
                           "(NOT tag1052720 IS NULL AND tag1052720 != 'None' AND tag1052720 != '' AND CAST(tag1052720 AS REAL) > 30) AND "
                           "(NOT tag1572944 IS NULL AND tag1572944 != 'None' AND tag1572944 != '') AND "
                           "(NOT tag1572999 IS NULL AND tag1572999 != 'None'AND tag1572999 != '') AND "
                           "(NOT tag1576992 IS NULL AND tag1576992 != 'None' AND tag1576992 != '') AND "
                           "(NOT tag2621488 IS NULL AND tag2621488 != 'None' AND tag2621488 != '')"
                           ";")

    selection = mdb.select("SELECT UID FROM tbl_data;")

    for i in range(len(selection)):
        selection_dcm = mdb.get_data("UID = '" + selection[i][0] + "'")

        sd = selection_dcm[0]["SeriesDescription"].value.lower()

        if "sasha" in sd:
            tool_general.save_dcm(selection_dcm[0], os.path.join(path_out, "SASHA"))
        elif "shmolli" in sd:
            tool_general.save_dcm(selection_dcm[0], os.path.join(path_out, "ShMOLLI"))
        else:
            tool_general.save_dcm(selection_dcm[0], os.path.join(path_out, "MOLLI"))

        print(str(i+1) + " done", end="\r")

if False:
    path_in = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw_sorted_NoAnonym"
    path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw_sorted_NoAnonym_split"

    for subdir in os.listdir(path_in):
        path_dir = os.path.join(path_in, subdir)

        for subject in os.listdir(path_dir):
            if sum(len(files) for _, _, files in os.walk(os.path.join(path_dir, subject))) == 1:
                shutil.copytree(os.path.join(path_dir, subject), os.path.join(path_dir, subject).replace(path_in, os.path.join(path_out, "single_map")))
            elif sum(len(files) for _, _, files in os.walk(os.path.join(path_dir, subject))) > 1:
                shutil.copytree(os.path.join(path_dir, subject), os.path.join(path_dir, subject).replace(path_in, os.path.join(path_out, "multiple_maps")))

if True:
    import pydicom
    path_orig = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw\3DCSAG002\0031_preMOLLI533saxMOCOT1\1.3.6.1.4.1.53684.1.1.4.0.543.1613579949.124308.dcm"
    path_out = r"C:\Users\CMRT\Desktop\Test"
    dcm = pydicom.dcmread(path_orig)

    p1 = dcm.preamble
    p2 = dcm.file_meta.to_json()
    p3 = dcm.to_json()

    dcm_new = pydicom.dataset.Dataset.from_json(p3)
    dcm_new.ensure_file_meta()
    dcm_new.file_meta = pydicom.dataset.FileMetaDataset.from_json(p2)
    dcm_new.preamble = p1

    tool_general.save_dcm(dcm_new, path_out)

    for root, _,  files in os.walk(path_out):
        for file in files:
            dcm_two = pydicom.dcmread(os.path.join(root, file))

    bytes(str(p1, "utf-8"), "utf-8")
    #path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw_sorted_NoAnonym\MOLLI\3DCSAG002\0031_preMOLLI533saxMOCOT1\1.3.6.1.4.1.53684.1.1.4.0.543.1613579949.124308.dcm"
    #dcm = pydicom.dcmread(path, force=True)
    a = 0