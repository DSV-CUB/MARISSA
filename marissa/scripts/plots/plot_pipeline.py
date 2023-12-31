from marissa.modules.database import marissadb
from marissa.toolbox.tools import tool_statistics, tool_pydicom, tool_plot
from marissa.scripts.plots import basic_functions
from marissa.modules.regression import linear
import numpy as np
import os
import copy
from matplotlib import pyplot as plt
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

########################################################################################################################
# USER INPUT ###########################################################################################################
########################################################################################################################

project_path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\4 - Tools\marissa\appdata\projects\DoktorarbeitDSV.marissadb"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\1 - Project Documents\Pipeline"

parameters = ["PatientsAge", "PatientsSex", "System", "T1Sequence"]
parameter_name = ["age", "sex", "system", "sequence"]
reference = ["18Y", "M", "SIEMENS#Verio#syngo MR B17", "MOLLI"]
reference_str = ["18.0", "M", "SIEMENS#Verio#syngo MR B17", "MOLLI"]

colours = ["#008000", "#ff8000", "#ff0000"]
cohorts = ["Healthy", "HCM", "Amyloidosis"]


########################################################################################################################
# PREPARATION ##########################################################################################################
########################################################################################################################

project = marissadb.Module(project_path)
setupID = basic_functions.get_BPSP_setupID(project, path_training)
pids = basic_functions.get_pids(project, parameters)

data, info_data = basic_functions.get_data_info(project, "TRAINING", pids)
data_test, info_data_test = basic_functions.get_data_info(project, "TESTHEALTHY", pids)
data_patient_hcm, info_data_patient_hcm = basic_functions.get_data_info(project, "TESTPATIENTHCM", pids)
data_patient_amy, info_data_patient_amy = basic_functions.get_data_info(project, "TESTPATIENTAMYLOIDOSE", pids)

########################################################################################################################
# MR AND MASK IMAGE ####################################################################################################
########################################################################################################################
idx_image = 0
dcm = project.get_data(data[idx_image][0])[0]
pd = tool_pydicom.get_dcm_pixel_data(dcm, rescale=True)
mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(data[idx_image][1]))[0][0]
cmask = tool_plot.mask2rgba(mask, "#ffffff00", "#66006688")

index_segmented = np.argwhere(mask)
minx = np.min(index_segmented[:,0])
maxx = np.max(index_segmented[:,0])
miny = np.min(index_segmented[:,1])
maxy = np.max(index_segmented[:,1])

diffx = maxx - minx
diffy = maxy - miny

spread = np.max([diffx, diffy])

xmin = int(minx - np.ceil((spread-diffx)/2) - 5)
xmax = int(maxx + np.floor((spread-diffx)/2) + 5)
ymin = int(miny - np.ceil((spread-diffy)/2) - 5)
ymax = int(maxy + np.floor((spread-diffy)/2) + 5)

pd = pd[xmin:xmax+1, ymin:ymax+1]
cmask = cmask[xmin:xmax+1, ymin:ymax+1]

fig, ax = plt.subplots(1,1, figsize=(20, 20))
ax.imshow(pd, cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path_out, "input_image.jpg"), dpi=300)

fig, ax = plt.subplots(1,1, figsize=(20, 20))
ax.imshow(cmask)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path_out, "input_mask.jpg"), dpi=300)

fig, ax = plt.subplots(1,1, figsize=(20, 20))
ax.imshow(pd, cmap="gray")
ax.imshow(cmask)
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(path_out, "input_image_and_mask.jpg"), dpi=300)

########################################################################################################################
# HISTOGRAMS ###########################################################################################################
########################################################################################################################
ct = project.select("SELECT clustertype FROM tbl_setup WHERE setupID = " + str(setupID))[0][0]
bins = project.select("SELECT bins FROM tbl_setup WHERE setupID = " + str(setupID))[0][0]

md = project.get_standardization(dcm, mask, setupID)

exec("from marissa.modules.clustering import " + ct + " as clustering")
cm = clustering.Model(bins=bins)
_, cl = cm.run(md.value_progression[0], return_indeces=True)
bins=np.histogram(md.value_progression[0], bins=100)[1]

fig, ax = plt.subplots(1,1, figsize=(10, 10))
for i in range(np.max(cl)+1):
    binvalues = md.value_progression[0][cl == int(i)]
    ax.hist(binvalues, bins=bins, facecolor="#660066" + str(int(88 - 44 * (i / np.max(cl)))), label="bin " + str(i+1))
ax.legend()
ax.set_ylabel("T1 value frequency")
ax.set_xlabel("T1 [ms]")
plt.tight_layout()
plt.savefig(os.path.join(path_out, "input_histogram.jpg"), dpi=300)



fig, ax = plt.subplots(1,1, figsize=(10, 10))
for i in range(np.max(cl)+1):
    binvalues = md.value_progression[0][cl == int(i)]
    ax.hist(binvalues, bins=bins, facecolor="#0000ff" + str(int(88 - 44 * (i / np.max(cl)))), label="original bin " + str(i+1))
    binvalues = md.value_progression[-1][cl == int(i)]
    ax.hist(binvalues, bins=bins, facecolor="#008800" + str(int(88 - 44 * (i / np.max(cl)))), label="standardized bin " + str(i+1))
ax.axvline(np.mean(md.value_progression[0]), c="#0000ff", ls="--", lw=2, label="original mean")
ax.axvline(np.mean(md.value_progression[-1]), c="#008800", ls="--", lw=2, label="standardized mean")

diff = (np.mean(md.value_progression[-1]) - np.mean(md.value_progression[0]))
start = np.mean(md.value_progression[0]) + 0.1*diff
end = np.mean(md.value_progression[-1]) - 0.1*diff
diff = 0.7*diff

if np.mean(md.value_progression[-1]) - np.mean(md.value_progression[0]) > 0:
    ax.arrow(start, 0.9*np.max(ax.get_ylim()), diff, 0, color="#008800", head_width=0.02*np.max(ax.get_ylim()), head_length=0.1/0.7*diff, lw=2)
else:
    ax.arrow(start, 0.9*np.max(ax.get_ylim()), diff, 0, color="#008800", head_width=0.02*np.max(ax.get_ylim()), head_length=-0.1/0.7*diff, lw=2)

from matplotlib.colors import ListedColormap
num88 = tool_plot.convert_hex_to_int("88")[0]
num66 = tool_plot.convert_hex_to_int("ff")[0]
new_cm = []
for i in range(256):
    new_cm.append([0, (num88 * (i /255))/256, (num66 - (i /255 * num66))/256, 1])
new_cm = ListedColormap(new_cm)

diff = 0.6/0.7 * diff
for i in range(256):
    plt.plot([start+(i*diff/256), start+((i+1)*diff/256)], [0.9*np.max(ax.get_ylim()), 0.9*np.max(ax.get_ylim())], c=new_cm(i/256), lw=2)

ax.legend(fontsize=16, loc="upper left")
ax.set_ylabel("T1 value frequency", fontsize=14)
ax.set_xlabel("T1 [ms]", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(path_out, "output_histogram.jpg"), dpi=300)

########################################################################################################################
# TRAIN FUNCTION AGE ###################################################################################################
########################################################################################################################
for i in range(len(parameters)):
    if "age" in parameters[i].lower():
        break

pid_age = pids[i]
selection = project.select("SELECT SOPinstanceUID, segmentationID FROM tbl_standardization_match_data_setup_parameter WHERE setupID = '" + str(setupID) + "' AND parameterID = '" + str(pid_age) + "'")
bins = project.select("SELECT bins FROM tbl_setup WHERE setupID = " + str(setupID))[0][0]
ytype = project.select("SELECT ytype FROM tbl_setup WHERE setupID = " + str(setupID))[0][0]

x = [[] for _ in range(bins)]
y = [[] for _ in range(bins)]

for i in range(len(selection)):
    dcm = project.get_data(selection[i][0])[0]
    mask = project.select("SELECT mask FROM tbl_segmentation WHERE segmentationID = " + str(selection[i][1]))[0][0]
    md = project.get_standardization(dcm, mask, setupID)
    values = md.value_progression[0]
    _, cl = cm.run(values, return_indeces=True)
    for j in np.unique(cl):
        y[j] = y[j] + np.array(values)[cl==j].tolist()
        x[j] = x[j] + [project.get_data_parameters(selection[i][0], [pid_age])[0]] * len(np.array(values)[cl==j])

for i in range(bins):
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    #ax.scatter(x[i], y[i], c="#660066" + str(int(88 - 44 * (i / bins))), label="train data")

    indeces = np.argwhere(np.array(x[i]).astype(float)==float(reference_str[0])).flatten()
    y_ref = np.mean(np.array(y[i])[indeces])

    if ytype == "absolute":
        dy = y[i]-y_ref
    else:
        dy = 100 * (y[i] - y_ref) / y_ref

    ax.scatter(x[i], dy, c="#660066" + str(int(88 - 44 * (i / bins))), label="train data")

    regressor = linear.Model(ytype=ytype, load=project.select("SELECT regressor FROM tbl_standardization WHERE setupID = '" + str(setupID) + "' AND parameterID = '" + str(pid_age) + "' AND bin = " + str(i+1))[0][0])

    m = regressor.regression.coef_[0]
    n = regressor.regression.intercept_

    ax.plot([np.min(x[i])-5, np.max(x[i])+5], [regressor.predict(np.array(np.min(x[i])-5).reshape((1,1))), regressor.predict(np.array(np.max(x[i])+5).reshape((1,1)))], c="blue", lw=2, label="regression")
    #ax.axvline(float(reference_str[0]), c="#008800", lw=2, ls="--", label="reference")

    if ytype=="absolute":
        ax.set_ylabel("\u0394T1 [ms]")
    else:
        ax.set_ylabel("\u0394T1 [%]")

    ax.set_xlim([np.min(x[i])-5, np.max(x[i])+5])
    ax.set_xlabel("age [y]")
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, "regression_bin" + str(i+1) + ".jpg"), dpi=300)

a = 0

########################################################################################################################
# OUTPUT HIST ##########################################################################################################
########################################################################################################################
# see above