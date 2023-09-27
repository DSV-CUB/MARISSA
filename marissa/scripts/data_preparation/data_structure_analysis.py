import numpy as np
import os
from marissa.modules.database import marissadb
from marissa.modules.pesf import pesf
from marissa.toolbox.tools import tool_general, tool_R

path_db = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\DB"
path_data = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\3 - Measurements\T1Map_Raw"
mdb = marissadb.Module(path=path_db)
#mdb.import_data(path_data)

#mdb.plot_database_structure(r"D:\ECRC_AG_CMR\3 - Promotion\Project MARISSA\1 - Project Documents\Images")
#mdb.import_data_contours(path_data, "DZHK")

selection = mdb.select("SELECT tag524400, IIF(tag528446 like '%ShMOLLI%', 'ShMOLLI', IIF(tag528446 like '%SASHA%', 'SASHA', 'MOLLI')), "
                       "tag528528, tag1048640, SUBSTR(tag1052688,1,3), IIF(CAST(tag1052704 AS REAL) > 2.5, CAST(tag1052704 AS REAL)/100, CAST(tag1052704 AS REAL)), tag1052720, tag1572944, tag1572999, tag1576992, "
                       "SUBSTR(tag2621488,2,4) FROM tbl_data WHERE "
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

selection_data = np.zeros((len(selection), len(selection[0]))).astype(str)
for i in range(len(selection)):
    for j in range(len(selection[i])):
        selection_data[i,j] = selection[i][j]

data_list = []
for i in range(np.shape(selection_data)[1]):
    try:
        data = np.round(selection_data[:,i].flatten().astype(float), 2)
    except:
        data = selection_data[:,i].flatten().astype(str)
    data_list.append(data.tolist())
data_list = np.transpose(np.array(data_list))
columnnames = ["Manufacturer", "Sequence", "System", "Gender", "Age", "Height", "Weight", "ST", "FieldStrength", "SWVersion", "PixelSpacing"]

FAMD = tool_R.Setup_FAMD()
FAMD.run(data_list, os.path.join(path_db, "FAMD_Raw"), columnnames=columnnames)

data_list_combine = []
data_list_combine.append([s[0] + "#" + s[1] + "#" + s[2] for s in data_list[:,[0,2,9]]])
data_list_combine.append(data_list[:,1].tolist())
data_list_combine.append(data_list[:,3].tolist())
data_list_combine.append(data_list[:,4].tolist())
data_list_combine.append(np.round((data_list[:,6].astype(float) / (data_list[:,5].astype(float)**2)),2).tolist())
data_list_combine.append(data_list[:,8].tolist())
data_list_combine.append((data_list[:,7].astype(float) * data_list[:,10].astype(float) ** 2).tolist())
data_list_combine = np.transpose(np.array(data_list_combine))
columnnames_combine = ["System", "Sequence", "Gender", "Age", "BMI", "FieldStrength", "Voxel Size"]

#FAMD = tool_R.Setup_FAMD()
FAMD.run(data_list_combine, os.path.join(path_db, "FAMD_TagCombine"),columnnames=columnnames_combine)

a=0