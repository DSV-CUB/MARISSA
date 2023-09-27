import numpy as np
import os
import matplotlib.pyplot as plt

path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL\train_step2_top3_binning.txt"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL"
original_data_cov = 12.47

with open(path, "r") as file:
    read = file.readlines()
    file.close()

read = np.array([a.replace("\n", "").split("\t") for a in read])[1:,:]
setting = [read[i,3] + "#" + read[i,4] + "#" + read[i, 5] for i in range(len(read))]
setting_index = np.zeros((len(setting), 1)).flatten().astype(int)
us = np.unique(setting)

fig, ax = plt.subplots(figsize=(15, 10))

for c in range(len(us)):
    indeces = np.argwhere(np.array(setting)==us[c])
    setting_index[indeces] = c

settings = np.max(setting_index)

for i in range(settings):
    indeces = np.sort(np.argwhere(setting_index==i).flatten())

    cov1 = np.array(read[indeces[0],-1]).astype(float)

    c = np.array(read[indeces[1:],1]).astype(int)
    covs = np.array(read[indeces[1:],-1]).astype(float)

    ax.scatter([1], [cov1], c="blue", s=10, zorder=10)

    max_c = np.inf
    for j in range(len(c)):
        if c[j] < max_c:
            ax.plot([1, c[j]], [cov1, covs[j]], c="#66666688", ls="--")
            ax.scatter([c[j]], [covs[j]], c="blue", s=10, zorder=10)
        else:
            ax.plot([c[j-1], c[j]], [covs[j-1], covs[j]], c="#66666688", ls="--")
            ax.scatter([c[j]], [covs[j]], c="blue", s=10, zorder=10)
        max_c=c[j]

ax.axhline(original_data_cov, c="#660066", ls="--", zorder=2, label="original (non-standardized)")

optimum = read[np.argmin(np.array(read[:,-1]).astype(float))]
#ax.scatter([int(optimum[1])], [float(optimum[-1])], c="blue", s=8, zorder=11)
ax.scatter([int(optimum[1])], [float(optimum[-1])], edgecolor="#008800", facecolor="None", lw=3, s=80, zorder=11)
ax.axhline(float(optimum[-1]), c="#008800", ls="--", zorder=2, label="optimal standardization")

ax.set_ylabel("coefficient of variation [%]")
ax.set_xlabel("number of bins")

cmin = np.min(np.array(read[:,1]).astype(int))
cmax = np.max(np.array(read[:,1]).astype(int))
ax.set_xlim(cmin-0.5, cmax+0.5)
ax.set_yticks(np.arange(int(np.ceil(ax.get_ylim()[0])), int(np.floor(ax.get_ylim()[1]))+1, 1).tolist() + [original_data_cov, float(optimum[-1])])

for i in range(len(ax.yaxis.get_ticklabels())):
    if i == len(ax.yaxis.get_ticklabels()) - 1:
        ax.yaxis.get_ticklabels()[i].set_color("#008800")
    elif i == len(ax.yaxis.get_ticklabels()) -2:
        ax.yaxis.get_ticklabels()[i].set_color("#660066")
    else:
        pass


np.arange(1,3,1)

ax.set_xticks(np.arange(cmin, cmax+1, 1))
ax.grid(lw=1,c="#eeeeee")
ax.legend(loc="upper left")
#ax.set_ylim([0, np.max(np.array(read[:,-1]).astype(float)) + 1])

plt.savefig(os.path.join(path_out, "COV_progression.jpg"), dpi=300)

a = 0