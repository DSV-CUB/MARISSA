import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

path = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER\train_step2_top3_binning.txt"
path_out = r"C:\Users\CMRT\Documents\DSV\3 - Promotion\Project MARISSA\6 - Analysis\TRAINING_FINAL_PAPER"
original_data_cov = 12.47

with open(path, "r") as file:
    read = file.readlines()
    file.close()

read = np.array([a.replace("\n", "").split("\t") for a in read])[1:,:]
setting = [read[i,3] + "#" + read[i,4] + "#" + read[i, 5] for i in range(len(read))]
setting_index = np.zeros((len(setting), 1)).flatten().astype(int)
us = np.unique(setting)
clustering_method = read[:,2]
ucm = np.unique(clustering_method)

cmap = cm.get_cmap('turbo')
cmap2 = cm.get_cmap('Set2')

fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios': [2,1], 'height_ratios': [1]}, figsize=(15,10))

for c in range(len(us)):
    indeces = np.argwhere(np.array(setting)==us[c])
    setting_index[indeces] = c

settings = np.max(setting_index)+1

for i in range(settings):
    indeces = np.sort(np.argwhere(setting_index==i).flatten())

    current_setting = us[i]
    scatter_color = cmap(abs(i/(settings-1)))

    cov1 = np.array(read[indeces[0],-1]).astype(float)

    c = np.array(read[indeces[1:],1]).astype(int)
    covs = np.array(read[indeces[1:],-1]).astype(float)

    ax[0].scatter([1], [cov1], c=scatter_color, s=10, zorder=10)

    max_c = np.inf
    for j in range(len(c)):
        line_color = cmap2(np.argwhere(ucm==clustering_method[indeces[j+1]]).flatten()[0]/(len(ucm)))
        if c[j] < max_c:
            ax[0].plot([1, c[j]], [cov1, covs[j]], c=line_color, ls="--")
            ax[0].scatter([c[j]], [covs[j]], c=scatter_color, s=10, zorder=10)
        else:
            ax[0].plot([c[j-1], c[j]], [covs[j-1], covs[j]], c=line_color, ls="--")
            ax[0].scatter([c[j]], [covs[j]], c=scatter_color, s=10, zorder=10)
        max_c=c[j]

ax[0].axhline(original_data_cov, c="#660066", ls="-", zorder=2, label="original (non-standardized)")

optimum = read[np.argmin(np.array(read[:,-1]).astype(float))]
#ax.scatter([int(optimum[1])], [float(optimum[-1])], c="blue", s=8, zorder=11)
ax[0].scatter([int(optimum[1])], [float(optimum[-1])], edgecolor="#008800", facecolor="None", lw=3, s=80, zorder=11)
ax[0].axhline(float(optimum[-1]), c="#008800", ls="-", zorder=2, label="optimal standardization")

ax[0].set_ylabel("coefficient of variation [%]", fontsize=14)
ax[0].set_xlabel("number of bins", fontsize=14)

cmin = np.min(np.array(read[:,1]).astype(int))
cmax = np.max(np.array(read[:,1]).astype(int))
ax[0].set_xlim(cmin-0.5, cmax+0.5)
ax[0].set_yticks(np.arange(int(np.ceil(ax[0].get_ylim()[0])), int(np.floor(ax[0].get_ylim()[1]))+1, 1).tolist() + [original_data_cov, float(optimum[-1])])

for i in range(len(ax[0].yaxis.get_ticklabels())):
    if i == len(ax[0].yaxis.get_ticklabels()) - 1:
        ax[0].yaxis.get_ticklabels()[i].set_color("#008800")
    elif i == len(ax[0].yaxis.get_ticklabels()) -2:
        ax[0].yaxis.get_ticklabels()[i].set_color("#660066")
    else:
        pass

ax[0].set_xticks(np.arange(cmin, cmax+1, 1))
ax[0].xaxis.set_tick_params(labelsize=12)
ax[0].yaxis.set_tick_params(labelsize=12)
#ax[0].grid(lw=1,c="#eeeeee")

r1 = plt.Rectangle((cmin-0.5, ax[0].get_ylim()[0]), cmax-cmin+1, original_data_cov-ax[0].get_ylim()[0], fill=True, color="#00880011", zorder=1)
ax[0].add_patch(r1)
r1 = plt.Rectangle((cmin-0.5, original_data_cov), cmax-cmin+1, ax[0].get_ylim()[1]-original_data_cov, fill=True, color="#ff000011", zorder=1)
ax[0].add_patch(r1)

ax[0].grid(lw=1, c="#ffffff")


#ax.set_ylim([0, np.max(np.array(read[:,-1]).astype(float)) + 1])


legend_elements = []
for i in range(settings):
    rgba = cmap(i/(settings-1))
    legend_elements.append(Line2D([0], [0], markeredgecolor=rgba, markerfacecolor=rgba, marker="o", linestyle="", label=us[i].replace("#", " | ").replace("linearsvr", "LSVR").replace("randomforest", "RFR").replace("extratrees", "ETR") ))

for i in range(len(ucm)):
    rgba = cmap2(i/(len(ucm)))
    legend_elements.append(Line2D([0], [0], color=rgba, lw=2, ls="--", label=ucm[i].replace("_", " ")))

legend_elements.append(Line2D([0], [0], color="#660066", lw=2, ls="-", label="original (non standardized)"))
legend_elements.append(Line2D([0], [0], color="#008800", lw=2, ls="-", label="optimal (BPSP standardized"))


#ax[-1, -1].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=20)
#ax[-1, -1].axis("off")
legend = ax[1].legend(handles=legend_elements, loc='center left', frameon=True, fontsize=12)
legend.get_frame().set_facecolor('#efefef66')
ax[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(path_out, "COV_progression.jpg"), dpi=300)

a = 0