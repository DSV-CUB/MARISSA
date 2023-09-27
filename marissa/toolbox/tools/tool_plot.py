import numpy as np
from matplotlib import pyplot as plt
from marissa.toolbox import tools
import os
from datetime import datetime
from PyQt5 import QtGui
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def convert_hex_to_int(hexstring):
    hexs = str(hexstring)
    if hexs.startswith("#"):
        hexs=hexs[1:]
    numc = int(len(hexs)/2)
    colors = []
    for i in range(numc):
        colors.append(int(hexs[2*i:2*i+2], 16))
    return colors

def mask2rgba(mask, color0="#dcdcdc00", color1="#00ff0088"):
    colors0 = convert_hex_to_int(color0)
    colors1 = convert_hex_to_int(color1)

    intmask = np.copy(mask).astype("int16")

    delta_R = np.copy(intmask)
    delta_G = np.copy(intmask)
    delta_B = np.copy(intmask)
    delta_A = np.copy(intmask)

    delta_R[intmask == 0] = colors0[0]
    delta_R[intmask == 1] = colors1[0]

    delta_G[intmask == 0] = colors0[1]
    delta_G[intmask == 1] = colors1[1]

    delta_B[intmask == 0] = colors0[2]
    delta_B[intmask == 1] = colors1[2]

    try:
        delta_A[intmask == 0] = colors0[3]
        delta_A[intmask == 1] = colors1[3]
    except:
        delta_A[:] = 255

    delta_RGBA = np.dstack((delta_R, delta_G, delta_B, delta_A))

    return delta_RGBA

def masks2delta2rgba(mask_truth, mask_predict, alpha=0.2):
    """

    :param mask_truth: maske n x m
    :param mask_predict: maske n x m
    :param alpha: transparency value
    :return: delta mask with numbers 2 = both mask true, 1 = only mask truth true, -1 = only mask predict true, 0 = both False
            delta_rgba mask with red for mask_predict only, blue for mask_truth only and green for both true
    """
    orig = np.copy(mask_truth).astype("int16")
    pred = np.copy(mask_predict).astype("int16")
    minus_orig_pred = np.subtract(orig, pred).astype("int16")
    plus_orig_pred = np.add(orig, pred).astype("int16")
    plus_orig_pred[plus_orig_pred<2] = 0
    delta = minus_orig_pred + plus_orig_pred # 0 = none, -1 = only pred, 1 = only orig, 2 = both

    delta_R = np.copy(delta)
    delta_G = np.copy(delta)
    delta_B = np.copy(delta)
    delta_A = np.copy(delta)

    delta_R[delta == 0] = 220
    delta_R[delta == 1] = 0
    delta_R[delta == -1] = 255
    delta_R[delta == 2] = 0
    delta_R = np.expand_dims(delta_R, axis=2)

    delta_G[delta == 0] = 220
    delta_G[delta == 1] = 0
    delta_G[delta == -1] = 0
    delta_G[delta == 2] = 255
    delta_G = np.expand_dims(delta_G, axis=2)

    delta_B[delta == 0] = 220
    delta_B[delta == 1] = 255
    delta_B[delta == -1] = 0
    delta_B[delta == 2] = 0
    delta_B = np.expand_dims(delta_B, axis=2)

    delta_A[delta == 0] = 0
    delta_A[delta != 0] = int(np.clip((alpha * 255), 0, 255))
    delta_A = np.expand_dims(delta_A, axis=2)

    delta_RGBA = np.concatenate((delta_R, delta_G, delta_B, delta_A), axis = 2)

    return delta, delta_RGBA

def plot_contour(image, contours, export=False, tight=True):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,1,1)

    ax1.grid(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax1.imshow(image, cmap="gray")
    for contour in contours:
        ax1.plot(contour[:,1], contour[:,0], c="red")

    fig.set_tight_layout(True)

    if type(export) == str:
        plt.savefig(os.path.join(export, "plot_contour_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
        plt.clf()
        plt.close(fig)
    elif export:
        plt.show()
    return ax1

def plot_contour_mask(image, contours, export=False):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.grid(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax2.grid(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    ax1.imshow(image, cmap="gray")
    for contour in contours:
        ax1.plot(contour[:,1], contour[:,0], c="red")

    ax2.imshow(image, cmap="gray")
    mask = tools.tool_general.contour2mask(contours, image)
    _, mask_rgba = tools.tool_plot.masks2delta2rgba(mask, mask)
    ax2.imshow(mask_rgba)

    if type(export) == str:
        plt.savefig(os.path.join(export, "plot_contour_mask_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
        plt.clf()
        plt.close(fig)
    elif export:
        plt.show()
    return ax1, ax2

def plot_contour_mask_compare(image, contours, export=False):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)

    ax1.grid(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax1.imshow(image, cmap="gray")
    for contour in contours:
        ax1.plot(contour[:,1], contour[:,0], c="red")

    from datetime import datetime
    start = datetime.now()
    mask = tools.tool_general.contour2mask(contours, image, mode="RASTERIZE")
    end = datetime.now()
    print(end-start)

    mask, c_new = tools.tool_general.get_lcc(mask, True)
    mask = tools.tool_general.contour2mask(c_new, image, mode="RASTERIZE")
    #mask = tools.tool_general.mask2contour2mask(mask, image)

    ax1.plot(c_new[0][:,1], c_new[0][:,0], c="blue")
    ax1.plot(c_new[1][:,1], c_new[1][:,0], c="green")
    #ax1.plot(c_new[2][:,1], c_new[2][:,0], c="orange")

    start = datetime.now()
    mask1 = tools.tool_general.contour2mask(contours, image, mode="NUMPY")
    end = datetime.now()
    print(end-start)

    _, mask_rgba = tools.tool_plot.masks2delta2rgba(mask, mask1)
    ax1.imshow(mask_rgba)

    if type(export) == str:
        plt.savefig(os.path.join(export, "plot_contour_mask_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
        plt.clf()
        plt.close(fig)
    elif export:
        plt.show()
    return ax1

def plot_mask_contour(image, mask, export=False):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.grid(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    ax2.grid(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    ax1.imshow(image, cmap="gray")
    ax1.imshow(tools.tool_plot.masks2delta2rgba(mask, mask))

    ax2.imshow(image, cmap="gray")
    contours = tools.tool_hadler.mask2contour(mask, 2, check=2) / 2
    for contour in contours:
        ax2.plot(contour[:,0], contour[:,1], c="red")

    if type(export) == str:
        plt.savefig(os.path.join(export, "plot_mask_contour_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
        plt.clf()
        plt.close(fig)
    elif export:
        plt.show()
    return ax1, ax2

def plot_masks(image, rgba_masks, titles, legend_handle=None, export=False, tight=True, axes=None):

    if axes is None:
        fig, new_axes = plt.subplots(1, len(rgba_masks), gridspec_kw={'width_ratios': [1] * len(rgba_masks)}, dpi=250, figsize=(len(rgba_masks)*10, 10))
        if len(rgba_masks) == 1:
            new_axes = [new_axes]
    else:
        new_axes = axes

    for i in range(len(rgba_masks)):
        if not titles is None:
            new_axes[i].set_title(titles[i])
        new_axes[i].grid(False)
        new_axes[i].xaxis.set_visible(False)
        new_axes[i].yaxis.set_visible(False)
        new_axes[i].imshow(image, cmap="gray")
        new_axes[i].imshow(rgba_masks[i])
        if not legend_handle is None:
            new_axes[i].legend(handles=legend_handle, loc="upper right")

    if axes is None:
        if tight:
            fig.set_tight_layout(True)

        if type(export) == str:
            plt.savefig(os.path.join(export, "plot_masks_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
            plt.clf()
            plt.close(fig)
        elif export:
            plt.show()
    return axes

def plot_image(image, export=False, tight=True):
    fig = plt.figure(1, figsize=(10, 10), dpi=250)

    ax = fig.add_subplot(1,1,1)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(image, cmap="gray")

    fig.set_tight_layout(True)

    if type(export) == str:
        plt.savefig(os.path.join(export, "plot_image_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
        plt.clf()
        plt.close(fig)
    elif export:
        plt.show()

    return ax

def plot_fig_to_qtlabel(fig, qtlabel):
    #from matplotlib.figure import Figure
    #fig = Figure()

    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    qt_height, qt_width = qtlabel.width(), qtlabel.height() #axis swapped in matplotlib

    ratio_width = qt_width / fig_width
    ratio_height = qt_height / fig_height

    if ratio_height < ratio_width:
        new_height = qt_height
        new_width = fig_width * ratio_height
    else:
        new_width = qt_width
        new_height = fig_height * ratio_width

    fig.set_size_inches((new_width / fig.dpi, new_height / fig.dpi))
    fig.set_canvas(FigureCanvas(fig))
    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1], QtGui.QImage.Format_ARGB32))
    pixmap = QtGui.QPixmap(qimage)
    qtlabel.setPixmap(pixmap)
    return

def plot_equivalence(data_tests, data_reference, tolerance=0.95, alpha=0.05, export=False, tight=True, **kwargs):
    from scipy import special

    xlabels = kwargs.get("xlabels", None)
    ylabel = kwargs.get("ylabel", None)

    if xlabels is None or len(xlabels) < len(data_tests):
        xlabels = range(len(data_tests) + 2)
    else:
        xlabels = [""] + xlabels + [""]

    reference_std_factor = np.sqrt(2) * special.erfinv(tolerance)
    reference_mean = np.mean(data_reference)
    reference_std = np.std(data_reference) * reference_std_factor

    alpha_c = alpha / len(data_tests)# bonferoni corrected alpha
    CI = (1 - 2*alpha_c)
    test_std_factor = np.sqrt(2) * special.erfinv(CI)

    test_mean = []
    test_std = []

    for i in range(len(data_tests)):
        test_mean.append(np.mean(data_tests[i].flatten()) - reference_mean)
        test_std.append(test_std_factor * np.std(data_tests[i].flatten()))


    fig = plt.figure(1, figsize=(10, 10), dpi=250)

    ax = fig.add_subplot(1,1,1)
    ax.grid(False)
    ax.barh(-reference_std, len(data_tests) + 1, 2*reference_std, align="edge", color=(1,0,0,0.1))
    ax.axhline(0, 0, len(data_tests) + 1, color=(1,0,0,0.25), linestyle=":")
    ax.axhline(reference_std, 0, len(data_tests) + 1, color=(1,0,0,0.25), linestyle="--")
    ax.axhline(-reference_std, 0, len(data_tests) + 1, color=(1,0,0,0.25), linestyle="--")


    for i in range(len(data_tests)):
        ax.errorbar(i+1, test_mean[i], yerr=test_std[i], capsize=2, color="blue")
        ax.scatter(i+1, test_mean[i], color="blue")

    ax.set_xlim(0, len(data_tests) + 1)
    ax.set_xticks(list(range(len(data_tests) + 2)))
    ax.set_xticklabels(xlabels)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize='x-small')

    if not ylabel is None:
        ax.set_ylabel(ylabel)

    fig.set_tight_layout(True)

    if type(export) == str:
        plt.savefig(os.path.join(export, "plot_image_" + datetime.now().strftime("%Y%m%d_%H%M%S%f") + ".jpg"))
        plt.clf()
        plt.close(fig)
    elif export:
        plt.show()

    return ax