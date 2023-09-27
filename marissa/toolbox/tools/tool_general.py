import numpy as np
import types
import pickle
import sys
import copy
import os
from osgeo import gdal

from skimage.transform import rescale, resize
from skimage.measure import label
#from skimage import morphology
from scipy.ndimage import rotate, morphology
from scipy import signal
from shapely.geometry import polygon as shapely_polygon, Point, MultiPoint
from rasterio.features import rasterize
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape


from marissa.toolbox.tools import tool_plot, tool_hadler

def save_dcm(dcm, path, overwrite=False):
    '''
    Saves Dicom data in a standardized directory structure
    :param dcm: pydicom object
    :param path: generous path to save the data
    :param overwrite: overwrite if already exists
    :return: path to store the data
    '''
    name = ''.join(e for e in str(dcm[0x0010, 0x0010].value) if e.isalnum())
    seriesnumber = str(dcm[0x0020, 0x0011].value).zfill(4)
    serieddescription = ''.join(e for e in str(dcm[0x0008, 0x103E].value) if e.isalnum())
    uid = str(dcm[0x0008, 0x0018].value)

    dir_path = os.path.join(path, name, seriesnumber + "_" + serieddescription)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, uid + ".dcm")
    if not overwrite and os.path.isfile(file_path):
        pass
    else:
        dcm.save_as(file_path)
    return dir_path

def distance(x, y, **kwargs):
    '''
    Calculates the Distance
    :param x: n x m matrix with n number of points and m dimensions
    :param y: n x m matrix with n number of points and m dimensions
    :param order: order of the distance calculation
    :return: distance for each row n between x and y (n x 1 matrix)
    '''
    order = kwargs.get("order", 2) # Minkowski Order, standard 2 (Euclidean)


    subtract = np.abs(np.subtract(x, y))
    result = np.linalg.norm(subtract, ord=order, axis=1)
    return result


def inner_distance(x, **kwargs):
    '''

    :param x: n x m matrix of points with n rows and m dimensions
    :return: matrix (distance of each point to each other), max(max occuring distance), min (min occuring distance)
    '''
    show_limits = kwargs.get("limits", False)

    x_n = np.tile(x, (len(x), 1))
    y = np.repeat(x, repeats=len(x), axis=0)

    dist = distance(x_n, y, **kwargs)

    matrix = np.reshape(dist, (len(x), len(x)))
    dist_min = np.min(dist[dist.nonzero()])
    dist_max = np.max(dist)

    if show_limits:
        return matrix, dist_min, dist_max
    else:
        return matrix

def outer_distance(x, y, **kwargs):
    '''

    :param x: n x m matrix of points with n rows and m dimensions
    :return: matrix (distance of each point to each other), max(max occuring distance), min (min occuring distance)
    '''

    show_limits = kwargs.get("limits", False)

    x_n = np.tile(x, (len(y), 1))
    y_n = np.repeat(y, repeats=len(x),axis=0)

    dist = distance(x_n, y_n, **kwargs)

    matrix = np.transpose(np.reshape(dist, (len(y), len(x))))
    dist_min = np.min(dist[dist.nonzero()])
    dist_max = np.max(dist)

    if show_limits:
        return matrix, dist_min, dist_max
    else:
        return matrix


def read_file(str_path, start=None, stop=None):
    file = open(str_path)
    lines = file.readlines()
    file.close()

    boo_read = False
    str_read = ''

    for num, line in enumerate(lines):
        #start of read
        if start is None or str(start) == str(num):
            boo_read = True
        elif line == str(start):
            boo_read = True
            continue

        # end of read
        if line == str(stop) or str(stop) == str(num):
            break

        # read if in block between start and end
        if boo_read:
            str_read = str_read + line

    return str_read


def read_file_and_split(str_path, start=None, stop=None, split_row="\n", split_column="\t"):
    str_read = read_file(str_path, start, stop)
    str_read = str_read.rstrip(split_row)
    return [element.split(split_column) for element in str_read.split(split_row)]

def string_stripper(string, allowed=[" ", "-", "_", "."]):
    return "".join([s for s in string if (s.isalnum() or s in allowed)]).strip()


def tf_gpu_set_memory_growth(set=True):
    import tensorflow as tf


    value = False
    #gpus = eval("tf.configuration.experimental.list_physical_devices('GPU')")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, set)
                #eval("tf.configuration.experimental.set_memory_growth(gpu, set)")
                value = True
        except:
            pass

    #probably alternative:
    #configuration = tf.ConfigProto()
    #configuration.gpu_options.allow_growth = True
    #sess = tf.Session(configuration=configuration)
    return value


def tf_copy_weights(model_source, model_target, final_layer=""):
    #import tensorflow as tf

    # https://medium.com/howtoai/model-surgery-copy-weights-from-model-to-model-a31b1dec7a7a
    for layer_target, layer_source in zip(model_target.layers,model_source.layers):
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        if layer_target.name == final_layer:
            break
    return True

def tf_clear_session():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    return True

def get_lcc(mask, connect_diagonal=False):

    if connect_diagonal:
        segmentation = np.kron(mask,np.ones((2, 2)))
        cfilter = np.array([[-1, 1], [1,-1]])
        conv = signal.convolve(segmentation, cfilter, mode="valid")
        conv = np.round(conv,0).astype(int)

        indeces = np.argwhere(conv==2)
        if len(indeces) > 0:
            indeces_1 = np.copy(indeces)
            indeces_2 = indeces + 1

        segmentation[indeces_1[:,0], indeces_1[:,1]] = 1
        segmentation[indeces_2[:,0], indeces_2[:,1]] = 1

        indeces = np.argwhere(conv==-2)
        if len(indeces) > 0:
            indeces_1 = np.transpose(np.vstack((indeces[:,0] + 1, indeces[:,1])))
            indeces_2 = np.transpose(np.vstack((indeces[:,0], indeces[:,1] + 1)))

        segmentation[indeces_1[:,0], indeces_1[:,1]] = 1
        segmentation[indeces_2[:,0], indeces_2[:,1]] = 1
    else:
        segmentation = copy.deepcopy(mask)


    if len(np.shape(segmentation)) == 3:
        lcc = []
        for i in range(np.shape(segmentation)[2]):
            labels = label(np.squeeze(segmentation[:, :, i]))
            if labels.max() == 0:
                lcc.append(segmentation)
            else:
                lcc.append(labels == np.argmax(np.bincount(labels.flat)[1:])+1)
        largestCC = np.array(lcc)
        largestCC = np.moveaxis(largestCC, 0, -1)
    else:
        labels = label(segmentation)
        if labels.max() == 0:
            largestCC = segmentation
        else:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    contourCC = mask2polygonalchain(largestCC)

    if connect_diagonal:
        for i in range(len(contourCC)):
            contourCC[i] = contourCC[i] / 2

        cfilter = np.array([[1, 1], [1,1]])
        largestCC = signal.convolve(largestCC, cfilter, mode="valid")
        largestCC = np.delete(largestCC, list(range(1, largestCC.shape[0], 2)), axis=0)
        largestCC = np.delete(largestCC, list(range(1, largestCC.shape[1], 2)), axis=1)
        largestCC = largestCC / 4
        largestCC[largestCC>0.5] = 1
        largestCC[largestCC<1] = 0


        #largestCC = largestCC[::2, ::2]
        #largestCC = array_resize(largestCC.astype(int), np.shape(mask), anti_aliasing=True, normalize=True)
        #largestCC[largestCC > 0.5] = 1
        #largestCC[largestCC < 1] = 0

    return largestCC, contourCC

def connected_components(segmentation, **kwargs):
    from skimage.measure import label
    seed = kwargs.get("seed", None)

    list_components = []
    index_largest_component=0
    labels = label(segmentation)
    if labels.max() > 0:
        indeces = (np.argsort(np.bincount(labels.flat)[1:])+1)[::-1]
        for i in indeces:
            if seed is None:
                list_components.append(np.argwhere(labels == i))
            elif [seed[0], seed[1]] in np.argwhere(labels == i).tolist():
                list_components.append(np.argwhere(labels == i))

    return list_components, index_largest_component

def connected_components_old(array, **kwargs):
    # breadth first search
    # https://link.springer.com/chapter/10.1007/978-1-4612-4400-4_4

    list_components = []
    index_largest_component = 0
    seed = kwargs.get("seed", None)

    value = array.flatten()
    lenx = np.shape(array)[0]
    leny = np.shape(array)[1]

    xx, yy = np.meshgrid(np.arange(0, lenx), np.arange(0, leny))

    x = np.transpose(xx).flatten()
    y = np.transpose(yy).flatten()

    group = np.zeros(len(value))
    group_max = 0

    for i in range(len(value)):
        if value[i] == 1:
            if group[i] == 0:
                group_max = group_max + 1
                group[i] = group_max
                current_group = group_max
            else:
                current_group = group[i]

            if i+1 < len(value) and value[i+1] == 1:
                next_group = group[i+1]
                if next_group == 0:
                    group[i+1] = current_group
                else:
                    group[group==next_group] = current_group

            if i+leny < len(value) and value[i+leny] == 1:
                next_group = group[i+leny]
                if next_group == 0:
                    group[i+leny] = current_group
                else:
                    group[group==next_group] = current_group

    unique_groups = np.unique(group)
    unique_groups = unique_groups[unique_groups.nonzero()]

    if seed is None:
        for i in range(len(unique_groups)):
            ix = x[group==unique_groups[i]]
            iy = y[group==unique_groups[i]]
            list_components.append(np.transpose([ix, iy]))
    else:
        seed_index = np.where((x == seed[0]) & (y == seed[1]))[0]
        seed_group = group[seed_index]

        if seed_group > 0:
            ix = x[group==seed_group]
            iy = y[group==seed_group]
            list_components.append(np.transpose([ix, iy]))
        else:
            list_components.append([])

    for i in range(len(list_components)-1):
        if len(list_components[index_largest_component]) < len(list_components[i+1]):
            index_largest_component = i+1

    return list_components, index_largest_component


def histogram(array, num_bins, range, normalize=True):
    data_hist, data_edges = np.histogram(array, bins=num_bins, range=range)
    if normalize:
        data_hist = data_hist / np.sum(data_hist) #np.linalg.norm(data_hist)
    data_max = np.max(array)
    return data_hist, data_edges, data_max


def get_cubic_spline_2D(arr_samplepoints, int_steps=100):
    # based on:
    # https://www-m2.ma.tum.de/foswiki/pub/M2/Allgemeines/NODE/musterloesung_2.pdf
    # http://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/16spline-curves.pdf
    # currently only natural boundary condition

    points = np.asarray(arr_samplepoints, 'float')

    if points.shape[1] != 2 or points.shape[0] < 3:
        raise TypeError(
            'points must be a n x 2 matrix with n >= 3, but the points given are of shape ' + str(points.shape))

    size = points.shape[0]
    t = np.arange(0, size, 1)
    M = np.zeros((4 * (size - 1), 4 * (size - 1)))
    x = np.zeros(4 * (size - 1))
    y = np.zeros(4 * (size - 1))

    x[0:size - 1] = points[0:size - 1, 0]
    x[size - 1:size + size - 2] = points[1:size, 0]

    y[0:size - 1] = points[0:size - 1, 1]
    y[size - 1:size + size - 2] = points[1:size, 1]

    tpow = np.ones((size, 4))
    tpow[:, 0] = np.power(t, 3)
    tpow[:, 1] = np.power(t, 2)
    tpow[:, 2] = t

    derivative_1 = np.array([3, 2, 1, 0])
    derivative_2 = np.array([6, 2, 0, 0])

    for i in range(0, size - 1):
        M[i, 4 * i:4 * i + 4] = tpow[i, :]

    for j in range(0, size - 1):
        M[i + j + 1, 4 * j:4 * j + 4] = tpow[j + 1, :]

    tpow = np.roll(tpow, -1, axis=1)

    k = 0
    for k in range(1, size - 1):
        M[i + j + 1 + k, 4 * (k - 1):4 * (k - 1) + 8] = np.concatenate(
            (np.multiply(derivative_1, tpow[k, :]), np.negative(np.multiply(derivative_1, tpow[k, :]))))

    tpow = np.roll(tpow, -1, axis=1)

    l = 0
    for l in range(1, size - 1):
        M[i + j + 1 + k + l, 4 * (l - 1):4 * (l - 1) + 8] = np.concatenate(
            (np.multiply(derivative_2, tpow[l, :]), np.negative(np.multiply(derivative_2, tpow[l, :]))))

    # if np.array_equal(points[0, :], points[size - 1, :]):
    #    print('ok')
    #    # natural condition if first and last are the same points
    #    # s1''(t0) = sn-1''(tn)
    #    M[i + j + 1 + k + l + 1, 0:4] = np.multiply(derivative_2, tpow[0, :])
    #    M[i + j + 1 + k + l + 1, 4 * (size - 2):] = np.negative(np.multiply(derivative_2, tpow[size - 1, :]))

    #    # s1'(t0) = sn-1'(tn)
    #    tpow = np.roll(tpow, 1, axis=1)
    #    M[i + j + 1 + k + l + 2, 0:4] = np.multiply(derivative_1, tpow[0, :])
    #    M[i + j + 1 + k + l + 2, 4 * (size - 2):] = np.negative(np.multiply(derivative_1, tpow[size - 1, :]))
    # else:
    #    # natural condition
    #    M[i + j + 1 + k + l + 1, 0:4] = np.multiply(derivative_2, tpow[0, :])
    #    M[i + j + 1 + k + l + 2, 4 * (size - 2):] = np.multiply(derivative_2, tpow[size - 1, :])

    M[i + j + 1 + k + l + 1, 0:4] = np.multiply(derivative_2, tpow[0, :])
    M[i + j + 1 + k + l + 2, 4 * (size - 2):] = np.multiply(derivative_2, tpow[size - 1, :])

    coeff_x = np.linalg.solve(M, x)
    coeff_y = np.linalg.solve(M, y)

    xy = np.zeros((int_steps * (size - 1), 2))
    t_fine = np.linspace(0, size - 1, (int_steps * (size - 1)))

    for i in range(0, size - 1):
        xy[int_steps * i:int_steps * i + int_steps, 0] = np.multiply(coeff_x[4 * i],
                                                                     np.power(t_fine[int_steps * i:int_steps * i + int_steps], 3)) + np.multiply(
            coeff_x[4 * i + 1], np.power(t_fine[int_steps * i:int_steps * i + int_steps], 2)) + np.multiply(coeff_x[4 * i + 2], t_fine[
                                                                                                                                int_steps * i:int_steps * i + int_steps]) + \
                                                         coeff_x[4 * i + 3]
        xy[int_steps * i:int_steps * i + int_steps, 1] = np.multiply(coeff_y[4 * i],
                                                                     np.power(t_fine[int_steps * i:int_steps * i + int_steps], 3)) + np.multiply(
            coeff_y[4 * i + 1], np.power(t_fine[int_steps * i:int_steps * i + int_steps], 2)) + np.multiply(coeff_y[4 * i + 2], t_fine[
                                                                                                                                int_steps * i:int_steps * i + int_steps]) + \
                                                         coeff_y[4 * i + 3]

    return xy


def polygonalchain2mask(polygonalchain, points, mode="RASTERIZE", on_line_inside=True):
    '''
    calculates the points inside a polygon
    attention here x has dimension 0 and y dimension 1 (like numpy arrays), but images have swaped columns, so polygon must be swaped afterwards
    :param polygonalchain: n x 2 (single polygon) or n x 4 (polygon sides)
    :param points: tuple (number of points in x and y direction) or m x n matrix of plane
    :param mode: mode of calculation: rasterize (fastest, but minimal divergence), shapely (slowest), numpy (Jordan Test)
    :param on_line_inside: in numpy mode one can choose if pixels with mid point on a polygonal side is referred as inside (True) or outside (False)
    :return: xy - array of indeces of points within segment, segmented_points - m x n matrix with ones at points that are inside (if points input has size 2, otherwise it is the masked points array [original value when inside]) and zeros outside, segmented_values - array of corresponding values
    '''
    # coordinate values of the points
    if np.size(points) == 2:
        rows = points[0]
        cols = points[1]
    else:
        rows = np.size(points, 0)
        cols = np.size(points, 1)

    # coordinates of each point
    #xy = np.transpose(np.reshape(np.meshgrid(np.arange(0, rows, 1), np.arange(0, cols, 1)), (2, rows * cols))) #+ 0.5

    # polygonalchain pieces start (1) and end (2) point
    if np.size(polygonalchain, 1) == 2:
        polygon1 = np.copy(polygonalchain)
        polygon2 = np.roll(polygonalchain, 1, 0)
    else:
        polygon1 = np.copy(polygonalchain[:, 0:2])
        polygon2 = np.copy(polygonalchain[:, 2:4])

    xmin = int(np.floor(np.min([np.min(polygon1[:, 0]), np.min(polygon2[:, 0])])))
    xmax = int(np.ceil(np.max([np.max(polygon1[:, 0]), np.max(polygon2[:, 0])])))
    ymin = int(np.floor(np.min([np.min(polygon1[:, 1]), np.min(polygon2[:, 1])])))
    ymax = int(np.ceil(np.max([np.max(polygon1[:, 1]), np.max(polygon2[:, 1])])))
    xy = np.transpose(np.reshape(np.meshgrid(np.arange(xmin, xmax+1, 1), np.arange(ymin, ymax+1, 1)), (2, -1)))  # + 0.5

    if mode.upper() == "RASTERIZE":

        polygon1 = polygon1 + 0.5
        rpoly = rasterize([shapely_polygon.Polygon(polygon1[:, [1, 0]])], (rows, cols))

        xy_int = np.argwhere(rpoly>0).astype(int) #+ 0.5 # center of each pixel


        if np.size(points) == 2:
            segmented_points = rpoly.astype(int)
            segmented_values = None
        else:
            segmented_points = np.zeros((rows, cols))
            segmented_points[xy_int[:, 0], xy_int[:, 1]] = points[xy_int[:, 0], xy_int[:, 1]]
            segmented_values = points[xy_int[:, 0], xy_int[:, 1]]


    else:
        if mode.upper() == "SHAPELY":
            sply_poly = shapely_polygon.Polygon(polygon1[:, [1, 0]])
            sply_poly_point = sply_poly.representative_point()
            inside = np.zeros(np.size(xy, 0))
            for i in range(0, np.size(xy, 0)):
                inside[i] = sply_poly.contains(MultiPoint([Point(xy[i,:]), sply_poly_point]))
        else:
            # start point is always above end point
            pol_diff = polygon1[:, 1] - polygon2[:, 1]
            copy_polygon1 = np.copy(polygon1)
            polygon1[pol_diff < 0, :] = polygon2[pol_diff < 0, :]
            polygon2[pol_diff < 0, :] = copy_polygon1[pol_diff < 0, :]

            # check if point is inside: Jordan Test
            inside = np.zeros(np.size(xy, 0))
            for i in range(0, np.size(xy, 0)):
                # flag all polygonalchain sides that are not affected
                no_cross = np.where((xy[i, 1] > polygon1[:, 1]) | (xy[i, 1] <= polygon2[:, 1]), 1, 0)

                # flag all polygonalchain sides where the point is right of the polygonalchain side (outerproduct = sin(a) Pol1 - point - Pol2)
                # Kreuzprodukt
                oprod = ((polygon1[:, 0] - xy[i, 0]) * (polygon2[:, 1] - xy[i, 1])) - ((polygon2[:, 0] - xy[i, 0]) * (polygon1[:, 1] - xy[i, 1]))

                # if point lies on any polygonalchain-side, it is determined as inside (if on_line_inside is True)
                if np.sum(np.where((no_cross == 0) & ((oprod == 0) & (((xy[i, 0] <= polygon1[:, 0]) & (xy[i, 0] >= polygon2[:, 0])) | ((xy[i, 0] >= polygon1[:, 0]) & (xy[i, 0] <= polygon2[:, 0])))), 1, 0)) > 0 and on_line_inside:
                    inside[i] = 1
                else:
                    oprod = np.where((oprod > 0), 1, 0)

                    # determine all relevant crossings
                    oprod = oprod - no_cross
                    oprod = np.where(oprod < 0, 0, oprod)

                    # calculate the number of crossings with the polygonalchain
                    total = np.sum(oprod)

                    # 1 if inside, 0 if not
                    inside[i] = np.mod(total, 2)

        xy = xy[inside == 1, :] #+ 0.5 # center of each pixel

        segmented_points = np.zeros((rows, cols))

        xy_int = xy.astype(int)

        if np.size(points) == 2:
            segmented_points[xy_int[:, 0], xy_int[:, 1]] = 1
            segmented_values = None
        else:
            segmented_points[xy_int[:, 0], xy_int[:, 1]] = points[xy_int[:, 0], xy_int[:, 1]]
            segmented_values = points[xy_int[:, 0], xy_int[:, 1]]

    return [xy_int, segmented_points, segmented_values]

def mask2polygonalchain(mask):

    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape

    rshape = shapes(np.transpose(mask).astype("uint8"))
    polygonalchain = []
    for vec in rshape:
        polygonalchain.append(np.array(shapely_shape(vec[0]).exterior.coords) - 1)

    return polygonalchain[:-1]

def mask2contour(mask):
    contours = mask2polygonalchain(mask_highres(mask))
    for i in range(len(contours)):
        contours[i] = contours[i] / 2
    return contours

def mask_highres(mask):
    high_res_mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    #mask_out = np.copy(high_res_mask)
    #for i in range(np.shape(high_res_mask)[0]-1):
    #    for j in range(np.shape(high_res_mask)[1]-1):
    #        part = high_res_mask[i:i+2,j:j+2]
    #        if np.array_equal(part.astype(int), np.array([1, 0, 1, 0]).astype(int).reshape((2,2))) or np.array_equal(part.astype(int), np.array([0, 1, 0, 1]).astype(int).reshape((2,2))):
    #            mask_out[i:i+2,j:j+2] = 1

    list_00 = np.argwhere(high_res_mask)
    list_10 = np.argwhere(high_res_mask) + np.array([1, 0])
    list_01 = np.argwhere(high_res_mask) + np.array([0, 1])
    list_11 = np.argwhere(high_res_mask) + np.array([1, 1])
    list_n10 = np.argwhere(high_res_mask) + np.array([-1, 0])
    list_n11 = np.argwhere(high_res_mask) + np.array([-1, 1])

    list_out = np.argwhere(high_res_mask)

    for i in range(len(list_00)):
        if any(np.equal(list_00, list_11[i]).all(1)) and not any(np.equal(list_00, list_10[i]).all(1)) and not any(np.equal(list_00, list_01[i]).all(1)):
            list_out = np.vstack([list_out, list_00[i,:] + np.array([1, 0])])
            list_out = np.vstack([list_out, list_00[i,:] + np.array([0, 1])])
        elif any(np.equal(list_00, list_n11[i]).all(1)) and not any(np.equal(list_00, list_n10[i]).all(1)) and not any(np.equal(list_00, list_01[i]).all(1)):
            list_out = np.vstack([list_out, list_00[i,:] + np.array([-1, 0])])
            list_out = np.vstack([list_out, list_00[i,:] + np.array([0, 1])])

    mask_out = np.zeros(np.shape(high_res_mask))
    mask_out[tuple(list_out.T)] = 1

    return mask_out

def classification_match(x, y=None):
    '''

    :param x: m x n x o if y is None, else m x 1 with m = dataset number, n = 2 (0=x/independent variable, 1=y/dependent variable), o = value(s)
    :param y: m x o if x is m x 1 -> dependent variable
    :return:
    '''
    if y is None:
        in_x = x[:][0][:]
        in_y = x[:][1][:]
    else:
        in_x = x
        in_y = y

    in_x = np.array(in_x)
    sorted_x = np.unique(in_x)
    sorted_y = []
    sorted_y_indeces = []
    for i in range(len(sorted_x)):
        indeces = np.argwhere(in_x == sorted_x[i])
        group_y = np.empty([0,0])
        group_y_indices = np.empty([0,0])
        for j in indeces:
            group_y = np.append(group_y, in_y[j[0]])
            group_y_indices = np.append(group_y_indices, j[0])
        sorted_y.append(group_y)
        sorted_y_indeces.append(group_y_indices)

    return sorted_x, sorted_y, sorted_y_indeces

def swap_order_213(x):
    new_x = []
    for j in range(len(x[0])):
        intermediate_x = []
        for i in range(len(x)):
            intermediate_x.append(x[i][j])
        new_x.append(intermediate_x)
    return new_x

def get_window_size():
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()

    if height >= 720:
        height = 720

    if width >= 1280:
        width = 1280
    else:
        width = (1280 / 720) * height

    stepsize = int((height * width) / (1280 * 720) * 25)

    return width, height, stepsize

def tk_center(tk_application):
    """
    centers a tkinter window
    :param win: the main window or Toplevel window to center
    """
    tk_application.update_idletasks()
    width = tk_application.winfo_width()
    frm_width = tk_application.winfo_rootx() - tk_application.winfo_x()
    win_width = width + 2 * frm_width
    height = tk_application.winfo_height()
    titlebar_height = tk_application.winfo_rooty() - tk_application.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = tk_application.winfo_screenwidth() // 2 - win_width // 2
    y = tk_application.winfo_screenheight() // 2 - win_height // 2
    tk_application.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    tk_application.deiconify()
    return True


def tk_button(master, command, **kwargs):
    import tkinter as tk

    c = kwargs.get("colour", "gray")
    text = kwargs.get("text", "Button")
    bw = kwargs.get("borderwidth", 0)

    colour_light = eval("configuration.colours.colour_hex_" + c + "_light")
    colour_dark = eval("configuration.colours.colour_hex_" + c + "_dark")

    return tk.Button(master, text=text, bg=colour_light, fg=colour_dark, activebackground=colour_dark, activeforeground=colour_light, command=command, borderwidth=bw)


def check_class_has_method(obj, method_name):
    return hasattr(obj, method_name) and type(getattr(obj, method_name)) == types.MethodType

def image_augmentation(input_array, exclude=None, contour=None, return_geometric_only=False):#img, mask=None):

    if contour is None or type(contour) == list:
        out_contour = copy.deepcopy(contour)
    else:
        out_contour = [contour]

    array = copy.deepcopy(input_array).astype("float32")
    if len(np.shape(array)) == 2:
        array = np.expand_dims(array, axis=2)
    min_val = np.min(array[:,:,0])
    max_val = np.max(array[:,:,0])

    if return_geometric_only:
        if len(np.shape(input_array)) == 2:
            out_geometry = np.expand_dims(copy.deepcopy(input_array).astype("float32"), axis=2)
        else:
            out_geometry = copy.deepcopy(input_array[:,:,0]).astype("float32")


    run = True
    counter = 0

    if exclude is None or exclude=="":
        exclude_list = []
    elif type(exclude) == list:
        exclude_list = exclude
    else:
        exclude_list = [exclude]

    while run and counter < 10:
        counter = counter + 1

        # brightness
        prob=np.random.randint(0, 100)
        if prob<25 and not "brightness" in exclude_list:
            excitation = 1 + (np.clip(np.random.normal(0.0, 0.388), -1, 1) * 0.5)
            array[:,:,0] = excitation * array[:,:,0]

        # contrast
        prob=np.random.randint(0, 100)
        if prob<25 and not "contrast" in exclude_list:
            change_val = max_val * (prob / 1000)
            prob=np.random.randint(0, 2)
            if prob < 1:
                change_val = -change_val
            array[:,:,0] = array[:,:,0] + change_val

        # blurring:
        prob=np.random.randint(0, 100)
        if prob < 25 and not "blurring" in exclude_list:
            k_power = float((abs(np.clip(np.random.normal(0.0, 0.388), -1, 1)) + 1) / 2)
            k_con = k_power * np.array([0.5, 1, 0.5])
            array[:,:,0] = np.apply_along_axis(lambda x: np.convolve(x, k_con, mode='same'), 0, array[:,:,0])
            array[:,:,0] = np.apply_along_axis(lambda x: np.convolve(x, k_con, mode='same'), 1, array[:,:,0])

        # add noise
        prob=np.random.randint(0, 100)
        if prob<25 and not "noise" in exclude_list:
            mode = np.random.randint(0,4)
            if mode < 3: #gaussian random noise
                std = abs(np.clip(np.random.normal(0.0, 0.388), -1, 1))
                if std != 0:
                    noise = np.clip(np.random.normal(0, std, np.shape(array[:,:,0])), -2, 2) * (max_val / 1000)
                    array[:,:,0] = array[:,:,0] + noise
            elif mode==3: # salt and pepper
                row,col,_ = np.shape(array)
                sp = 0.5
                amount = 0.0001 * (abs(np.clip(np.random.normal(0.0, 0.388), -1, 1)) + 1) * 50
                # Salt mode
                num_salt = np.ceil(amount * row * col * sp)
                coords_x = np.random.randint(0, row, int(num_salt))
                coords_y = np.random.randint(0, col, int(num_salt))
                array[coords_x, coords_y, 0] = np.max(array[:,:,0])

                # Pepper mode
                num_pepper = np.ceil(amount * row * col * (1 - sp))
                coords_x = np.random.randint(0, row, int(num_pepper))
                coords_y = np.random.randint(0, col, int(num_pepper))
                array[coords_x, coords_y, 0] = np.min(array[:,:,0])

        # rotation
        prob=np.random.randint(0, 100)
        if prob<25 and not "rotation" in exclude_list:
            angle = int(np.clip(np.random.normal(0.0, 0.388), -1, 1) * 180) # std = 0.388 means 99% are within -1 and 1
            array = rotate(array, angle, reshape=False, order=1)

            angle_rad = np.deg2rad(angle)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([[c, s],
                                      [-s, c]])
            out_plane_shape = np.asarray([array.shape[1], array.shape[0]])
            out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
            in_center = (np.copy(out_plane_shape) - 1) / 2
            offset = in_center - out_center

            if not contour is None:
                for i in range(len(out_contour)):
                    out_contour[i] = np.transpose(rot_matrix @ np.transpose(out_contour[i])[[1,0],:] + offset[:, np.newaxis])[:, [1,0]]

            if return_geometric_only:
                out_geometry = rotate(out_geometry, angle, reshape=False, order=1)

        # mirror
        prob=np.random.randint(0, 100)
        if prob<25 and not "mirror" in exclude_list:
            prob = np.random.randint(0, 3)
            if prob == 0 or prob == 2:
                array = np.flip(array, axis=0)

                if not contour is None:
                    for i in range(len(out_contour)):
                        out_contour[i][:,0] = out_contour[i][:,0] - 2 * (out_contour[i][:,0] - float(array.shape[0] / 2))

                if return_geometric_only:
                    out_geometry = np.flip(out_geometry, axis=0)

            if prob == 1 or prob == 2:
                array = np.flip(array, axis=1)

                if not contour is None:
                    for i in range(len(out_contour)):
                        out_contour[i][:,1] = out_contour[i][:,1] - 2 * (out_contour[i][:,1] - float(array.shape[1] / 2))

                if return_geometric_only:
                    out_geometry = np.flip(out_geometry, axis=1)

        # axis downsample
        prob=np.random.randint(0, 100)
        if prob<25 and not "downsample" in exclude_list:
            prob = np.random.randint(0, 3)

            if prob == 0 or prob == 2:
                ca = np.copy(np.squeeze(array[::2, :, 0]))
                ca = np.repeat(ca, 2, axis=0)
                ca = np.resize(ca, (np.shape(array)[0], np.shape(array)[1]))
                array[:, :, 0] = ca
            if prob == 1 or prob == 2:
                ca = np.copy(np.squeeze(array[:, ::2, 0]))
                ca = np.repeat(ca, 2, axis=1)
                ca = np.resize(ca, (np.shape(array)[0], np.shape(array)[1]))
                array[:, :, 0] = ca

        # crop
        prob=np.random.randint(0, 100)
        if prob<25 and not "crop" in exclude_list:
            size = 100 - int(abs(np.clip(np.random.normal(0.0, 0.388), -1, 1)) * 25)
            array = array[int((50-size/2) * np.shape(input_array)[0] / 100):int((50+size/2) * np.shape(input_array)[0] / 100), int((50-size/2) * np.shape(input_array)[1] / 100):int((50+size/2) * np.shape(input_array)[1] / 100),:]

            if not contour is None:
                for i in range(len(out_contour)):
                    out_contour[i][:,0] = out_contour[i][:,0] - int((50-size/2) * np.shape(input_array)[0] / 100)
                    out_contour[i][:,1] = out_contour[i][:,1] - int((50-size/2) * np.shape(input_array)[1] / 100)

            if return_geometric_only:
                out_geometry = out_geometry[int((50-size/2) * np.shape(input_array)[0] / 100):int((50+size/2) * np.shape(input_array)[0] / 100), int((50-size/2) * np.shape(input_array)[1] / 100):int((50+size/2) * np.shape(input_array)[1] / 100),:]

        array[:,:,0][array[:,:,0]>max_val] = max_val
        array[:,:,0][array[:,:,0]<min_val] = min_val

        if np.max(array[:,:,0]) < 1e-10 and max_val >= 1e-10:
            run = True
        else:
            run = False

    if len(np.shape(input_array)) == 2:
        array = np.squeeze(array)

    if return_geometric_only:
        out_geometry = np.squeeze(out_geometry)
        out_geometry[out_geometry>max_val] = max_val
        out_geometry[out_geometry<min_val] = min_val
        return array, out_contour, out_geometry
    else:
        return array, out_contour



def masks2rgba(mask1, mask2, **kwargs):
    alpha_value = kwargs.get("alpha", 0.5)

    orig = np.copy(mask1).astype("int16")
    pred = np.copy(mask2).astype("int16")

    minus_orig_pred = np.subtract(orig, pred).astype("int16")
    plus_orig_pred = np.add(orig, pred).astype("int16")
    plus_orig_pred[plus_orig_pred<2] = 0

    delta = minus_orig_pred + plus_orig_pred # 0 = none, -1 = only mask2, 1 = only mask1, 2 = both

    delta_R = np.copy(delta)
    delta_G = np.copy(delta)
    delta_B = np.copy(delta)

    delta_R[delta == 0] = 0
    delta_R[delta == 1] = 0
    delta_R[delta == -1] = 255
    delta_R[delta == 2] = 0
    delta_R = np.expand_dims(delta_R, axis=2)

    delta_G[delta == 0] = 0
    delta_G[delta == 1] = 0
    delta_G[delta == -1] = 0
    delta_G[delta == 2] = 255
    delta_G = np.expand_dims(delta_G, axis=2)

    delta_B[delta == 0] = 0
    delta_B[delta == 1] = 255
    delta_B[delta == -1] = 0
    delta_B[delta == 2] = 0
    delta_B = np.expand_dims(delta_B, axis=2)

    delta_RGB = np.concatenate((delta_R, delta_G, delta_B), axis=2) / 255

    alphas = np.amax(delta_RGB, axis=2)
    alphas[alphas>0] = alpha_value
    alphas = np.expand_dims(alphas, axis=2)

    delta_RGBA = np.concatenate((delta_RGB, alphas), axis=2)

    return delta_RGBA


def array_resize(array, new_size, **kwargs):
    anti_aliasing = kwargs.get("anti_aliasing", True)
    normalize = kwargs.get("normalize", False)

    shape = np.shape(array)
    if len(shape) > 2:
        new_array = np.reshape(array, (shape[0], shape[1], -1))
    else:
        new_array = np.expand_dims(array, axis=2)

    result=[]
    for j in range(np.shape(new_array)[-1]):
        resultj = eval(("resize" if isinstance(new_size, (list, tuple, np.ndarray)) else "rescale") + "(new_array[:,:,j].squeeze(), new_size, anti_aliasing=anti_aliasing)")
        resultjmax = np.amax(resultj, (0, 1))

        if not normalize or np.abs(resultjmax) == 0:
            result.append(resultj)
        else:
            result.append(resultj / resultjmax)

    result = np.moveaxis(np.array(result), 0, -1)

    if len(shape) == 2:
        result = result.squeeze()

    return result


def replace_right(source, target, replacement, replacements=None):
    return replacement.join(source.rsplit(target, replacements))


def get_metric_from_masks(mask1=None, mask2=None, metric="DSC", **kwargs):
    if mask1 is None or mask2 is None:
        result = ["DSC", "IOU", "HD", "ASD", "E", "AE", "SE", "E%"]
    elif metric.upper() == 'DSC' or metric.upper() == 'DICE':
        dscmask1, dscmask2 = mask1.astype(bool), mask2.astype(bool)
        numerator = 2 * np.sum(np.logical_and(dscmask1, dscmask2)) + 1
        denominator = np.sum(dscmask1) + np.sum(dscmask2) + 1
        result = 100 * (numerator / denominator)
    elif metric.upper() == "IOU":
        ioumask1, ioumask2 = mask1.astype(bool), mask2.astype(bool)
        numerator = np.sum(np.logical_and(ioumask1, ioumask2)) + 1
        denominator = np.sum(np.logical_or(ioumask1, ioumask2)) + 1
        result = 100 * (numerator / denominator)
    elif metric.upper() == "HD":
        voxel_sizes = kwargs.get("voxel_sizes", [1, 1])
        connectivity = kwargs.get("connectivity", 1)
        input_1 = np.atleast_1d(mask1.astype(np.bool8))
        input_2 = np.atleast_1d(mask2.astype(np.bool8))
        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
        S_1 = input_1 ^ morphology.binary_erosion(input_1, conn)
        S_2 = input_2 ^ morphology.binary_erosion(input_2, conn)
        dta = morphology.distance_transform_edt(~S_1, voxel_sizes)
        dtb = morphology.distance_transform_edt(~S_2, voxel_sizes)
        surface_distance = np.concatenate([np.ravel(dta[S_2!=0]), np.ravel(dtb[S_1!=0])])
        result = surface_distance.max()
    elif metric.upper() == "ASD":
        voxel_sizes = kwargs.get("voxel_sizes", [1, 1])
        connectivity = kwargs.get("connectivity", 1)
        input_1 = np.atleast_1d(mask1.astype(np.bool8))
        input_2 = np.atleast_1d(mask2.astype(np.bool8))
        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
        S_1 = input_1 ^ morphology.binary_erosion(input_1, conn)
        S_2 = input_2 ^ morphology.binary_erosion(input_2, conn)
        dta = morphology.distance_transform_edt(~S_1, voxel_sizes)
        dtb = morphology.distance_transform_edt(~S_2, voxel_sizes)
        surface_distance = np.concatenate([np.ravel(dta[S_2!=0]), np.ravel(dtb[S_1!=0])])
        result = surface_distance.mean()
    elif metric.upper() == "E":
        values = kwargs.get("values", np.zeros(np.shape(mask1)))
        indeces1 = np.argwhere(mask1)
        indeces2 = np.argwhere(mask2)

        if len(indeces1) > 0 and len(indeces2) > 0:
            values1 = values[indeces1[:,0], indeces1[:,1]]
            values2 = values[indeces2[:,0], indeces2[:,1]]
            result = (np.mean(values1) - np.mean(values2))
        elif len(indeces1) > 0:
            result = (np.mean(values[indeces1[:,0], indeces1[:,1]]))
        elif len(indeces2) > 0:
            result = (np.mean(values[indeces2[:,0], indeces2[:,1]]))
        else:
            result = 0
    elif metric.upper() == "AE":
        values = kwargs.get("values", np.zeros(np.shape(mask1)))
        indeces1 = np.argwhere(mask1)
        indeces2 = np.argwhere(mask2)

        if len(indeces1) > 0 and len(indeces2) > 0:
            values1 = values[indeces1[:,0], indeces1[:,1]]
            values2 = values[indeces2[:,0], indeces2[:,1]]
            result = np.abs(np.mean(values1) - np.mean(values2))
        elif len(indeces1) > 0:
            result = np.abs(np.mean(values[indeces1[:,0], indeces1[:,1]]))
        elif len(indeces2) > 0:
            result = np.abs(np.mean(values[indeces2[:,0], indeces2[:,1]]))
        else:
            result = 0
    elif metric.upper() == "SE":
        values = kwargs.get("values", np.zeros(np.shape(mask1)))
        indeces1 = np.argwhere(mask1)
        indeces2 = np.argwhere(mask2)

        if len(indeces1) > 0 and len(indeces2) > 0:
            values1 = values[indeces1[:,0], indeces1[:,1]]
            values2 = values[indeces2[:,0], indeces2[:,1]]
            result = (np.mean(values1) - np.mean(values2)) ** 2
        elif len(indeces1) > 0:
            result = (np.mean(values[indeces1[:,0], indeces1[:,1]])) ** 2
        elif len(indeces2) > 0:
            result = (np.mean(values[indeces2[:,0], indeces2[:,1]])) ** 2
        else:
            result = 0
    elif metric.upper() == "E%":
        values = kwargs.get("values", np.zeros(np.shape(mask1)))
        indeces1 = np.argwhere(mask1)
        indeces2 = np.argwhere(mask2)

        if len(indeces1) > 0 and len(indeces2) > 0:
            values1 = values[indeces1[:,0], indeces1[:,1]]
            values2 = values[indeces2[:,0], indeces2[:,1]]
            base = (np.mean(values1) + np.mean(values2)) / 2
            diff = np.abs((np.mean(values1) - np.mean(values2)) / 2)
            result = 100 * (diff / base)
        elif len(indeces1) > 0:
            result = 100
        elif len(indeces2) > 0:
            result = 100
        else:
            result = 0
    else:
        result = None

    if not result is None and not (mask1 is None or mask2 is None):
        result = round(result, 2)
    return result

def mask2contour2mask(mask, pixel_data, mode="RASTERIZE"):
    if np.max(mask.astype(int)) == 0:
        new_mask = np.zeros(np.shape(mask))
    else:
        try:
            spr = 4
            contour = tool_hadler.mask2contour(mask, 4, check=2)
            if len(contour) > 1:
                new_mask = polygonalchain2mask(get_cubic_spline_2D(contour[-1] / 4), np.shape(pixel_data), mode)[1] - polygonalchain2mask(get_cubic_spline_2D(contour[-2] / 4), np.shape(pixel_data), mode)[1]
            elif len(contour) == 1:
                new_mask = polygonalchain2mask(get_cubic_spline_2D(contour[-1] / 4), np.shape(pixel_data), mode)[1]
            else:
                new_mask = np.zeros(np.shape(mask))
        except:
            new_mask = np.zeros(np.shape(mask))
    return new_mask


def contour2mask(contour, pixel_data, cubic_spline=False, mode="RASTERIZE"):

    if cubic_spline:
        new_contour = []
        for i in range(len(contour)):
            new_contour.append(get_cubic_spline_2D(contour[i]))
    else:
        new_contour = copy.deepcopy(contour)

    if len(new_contour) > 1:
        new_mask = polygonalchain2mask(new_contour[-1], np.shape(pixel_data), mode)[1] - polygonalchain2mask(new_contour[-2], np.shape(pixel_data), mode)[1]
    else:
        new_mask = polygonalchain2mask(new_contour[-1], np.shape(pixel_data), mode)[1]

    return new_mask

def pickle_to_numpy(filepath, readmode="all"):
    file = open(filepath, "rb")
    read = pickle.load(file)
    file.close()

    mode = string_stripper(readmode).lower()
    result = []

    if mode == "all":
        for key in list(read.keys()):
            contour_points = read[key][0][:, [1,0]]
            x = read[key][2]
            y = read[key][1]

            if "points" in key.lower():
                mask = np.zeros((x, y))
                for i in range(len(contour_points)):
                    mask[int(np.round(contour_points[i,0])), int(np.round(contour_points[i,1]))] = 1
            else: # contour
                mask = contour2mask([contour_points], np.zeros((x, y)))
        result.append(mask)
    elif mode == "lvsax" and "saendocardialContour" in read and "saepicardialContour" in read:
        contour_points = [read["saendocardialContour"][0][:, [1,0]], read["saepicardialContour"][0][:, [1,0]]]
        x = read["saendocardialContour"][2]
        y = read["saendocardialContour"][1]
        mask = contour2mask(contour_points, np.zeros((x, y)))
        result.append(mask)
    return result





def array_resolution(array, factor):
    #from skimage.transform import resize
    #result = resize(array, (factor * np.asarray(np.shape(array))).astype(int))
    #import cv2

    from PIL import Image

    iarray = Image.fromarray(array.astype("float32"), "F")
    iarray = iarray.resize((factor * np.asarray(np.shape(array))).astype(int), Image.LINEAR) #cv2.resize(iarray, (factor * np.asarray(np.shape(array))).astype(int))
    result = np.asarray(iarray, dtype="float32").squeeze()


    return result

def rgb_to_greyscale(array):
    result = np.array(array)
    result = np.squeeze(result)

    if len(np.shape(result)) == 3 and np.shape(result)[-1] == 3:
        result = np.squeeze(0.3 * result[:,:,0] + 0.59 * result[:,:,1] + 0.11 * result[:,:,2])
    else:
        result = None

    return result

def get_image_sharpeness_measure(array):
    # https://reader.elsevier.com/reader/sd/pii/S1877705813016007?token=F90A4C91FF94E4B4AE6BDC51DF8556D4DE7B5E447696DECF5305888C9EB896F5C10020D4F1702E63E2293A165D3FB401&originRegion=eu-west-1&originCreation=20221130144050

    if len(np.shape(array)) == 3:
        image = rgb_to_greyscale(array)
    elif len(np.shape(array)) == 2:
        image = np.copy(array)
    else:
        image = None

    if not image is None:
        if not np.max(image) == 0:
            image = image / np.max(image)

        F = np.fft.fft2(image)
        Fc = np.fft.fftshift(F)
        AF = np.abs(Fc)
        M = np.max(AF)
        threshold = M / 1000
        Th = np.count_nonzero(F>threshold)
        FM = Th / (np.shape(image)[0] * np.shape(image)[1])
        result = FM
    else:
        result = None

    return result

def get_python_packages(path):
    packages = []
    versions = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(os.path.join(root, file), "r") as readfile:
                        readlines = readfile.readlines()
                    readfile.close()
                except:
                    continue

                for line in readlines:
                    sline = line.strip()
                    index_start = -1

                    if sline.startswith("import "):
                        index_start = 7

                        if "." in sline:
                            index_stop = sline.find(".")
                        elif " as " in sline:
                            index_stop = sline.find(" as ")
                        else:
                            index_stop = len(sline)

                    elif sline.startswith("from "):
                        index_start = 5

                        if "." in sline:
                            index_stop = sline.find(".")
                        elif " import " in sline:
                            index_stop = sline.find(" import ")
                        else:
                            index_stop = len(sline)

                    if index_start > -1:
                        package = sline[index_start:index_stop]
                        try:
                            exec("import " + package)
                            version = eval(package + ".__version__")
                        except:
                            version = ""

                        if not package in packages:
                            packages.append(package)
                            versions.append(version)
    return packages, versions

if __name__ == "__main__":
    pass