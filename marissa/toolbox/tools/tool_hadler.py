import copy
import os
import pickle
import numpy as np
import pydicom
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label, find_contours
from skimage.morphology import convex_hull_image
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection, Point, MultiPoint, shape
from shapely import affinity
from rasterio import features

def getLargestCC(segmentation):
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
    return largestCC


def convex_hull(segmentation):
    return convex_hull_image(segmentation)


def segm2contour(segmentation, threshold=0.5):
    return find_contours(segmentation, threshold)

def mask2contour(segmentation, subpixelresolution=2, check=None):
    if check is None:
        segm_lv1 = segmentation #[:,:,0]
        segm_lv1 = getLargestCC(segm_lv1)
        segm_lv1 = np.kron(segm_lv1,np.ones((subpixelresolution, subpixelresolution))) # upscaling for high res
        segm_lv1 = gaussian_filter(segm_lv1, sigma=2)  # better contours
        contours = sorted(find_contours(segm_lv1.squeeze(), 0.5), key=len)
    else:
        contours = sorted(find_contours(segmentation.squeeze(), 0.5), key=len)
        segm_lv1 = segmentation #[:,:,0]
        segm_lv1 = getLargestCC(segm_lv1)
        segm_lv1 = np.kron(segm_lv1,np.ones((subpixelresolution, subpixelresolution))) # upscaling for high res
        segm_lv1 = gaussian_filter(segm_lv1, sigma=2)  # better contours

        if len(contours) >= check:
            for i in range(50):
                contours = sorted(find_contours(segm_lv1.squeeze(), 0.5-i/100), key=len)
                if len(contours) >= check:
                    break
        else:
            contours = sorted(find_contours(segm_lv1.squeeze(), 0.5), key=len)
            
    return contours


#def from_polygon(polygon, shape):
#    mask = features.rasterize([polygon], shape=shape)
#    return mask

def from_polygon(polygons, shape, shift=0):
    """Convert to mask (Origin (0.0, 0.0))

    Note:
        rasterio.features.rasterize(shapes, out_shape=None, fill=0, out=None, transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        all_touched=False, merge_alg=MergeAlg.replace, default_value=1, dtype=None)
        For Origin (-0.5, -0.5) apply Affine Transformation (1.0, 0.0, -0.5, 0.0, 1.0, -0.5)
        https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html#rasterio.features.rasterize

    Args:
        polygons (shapely.geometry. Polygon | Multipolygon): geometry to be burned into mask
        height (int): output mask height
        width (int):  output mask width
        shift (int): shift mask by shift pixels (applied on both axis, necessary if there is a problem in coordinate transformation)

    Returns:
        ndarray (2D array of np.uint8): binarized mask of polygon
    """
    height, width = shape

    if not isinstance(polygons, list):
        if isinstance(polygons, Polygon) or isinstance(polygons, MultiPolygon):
            polygons = [polygons]
        else:
            raise Exception('from_polygon accepts a List of Polygons or Multipolygons')
    if len(polygons) > 0:
        try:
            mask = features.rasterize(polygons, out_shape=(height, width), dtype=np.uint8)
        except Exception as e:
            mask = np.zeros((height, width), np.uint8)
            print(str(e) + ', returning empty mask.')
    else:
        mask = np.zeros((height, width), np.uint8)

    if shift != 0:
        shift = int(np.round(shift,0))
        mask_new = np.zeros((np.shape(mask)[0]+abs(shift), np.shape(mask)[1]+abs(shift)))
        if shift > 0:
            mask_new[shift:, shift:] = mask
            mask_new = mask_new[0:-shift, 0:-shift]
        else: # shift < 0
            mask_new[:shift, :shift] = mask
            mask_new = mask_new[abs(shift):, abs(shift):]
        mask = mask_new.astype(np.uint8)
    return mask




def to_polygon(mask):
    """Convert mask to Polygons
    Args:    mask (ndarray (2D array of np.uint8): binary mask
    Returns: MultiPolygon | Polygon: Geometries extracted from mask, empty Polygon if empty mask
    """
    uintmask = np.array(mask).astype("uint8")

    polygons = []
    for geom, val in features.shapes(uintmask):
        if val:
            polygon = shape(geom)
            if polygon.geom_type == 'Polygon' and polygon.is_valid: polygons.append(polygon)
            else: print('Ignoring GeoJSON with cooresponding shape: ' + str(polygon.geom_type) + ' | Valid: ' + str(polygon.is_valid))
    return MultiPolygon(polygons) if len(polygons)>0 else Polygon()#polygons[0]


# In einem Ordner mit Namen des Readers alle Fälle des Readers packen
# Ordnername: StudyUID für den individuellen Fall
# Annotationsname des Pickle Files: SOPInstanceUID
def mask_to_LL_annotation(mask, dcm):
    mask_shape = np.shape(mask)
    dcm_shape = np.shape(dcm.pixel_array)
    scale_x = int(mask_shape[0] / dcm_shape[0])
    scale_y = int(mask_shape[1] / dcm_shape[1])
    if scale_x == scale_y:
        scale = scale_x
    else:
        raise RuntimeError("mask shape proportion does not fit dicom shape proportions.")

    myo = affinity.scale(to_polygon(mask), xfact=1./scale, yfact=1./scale, origin=(0,0))
    try:
        endo = Polygon(myo.geoms[0].interiors[0])
        epi = Polygon(myo.geoms[0].exterior)
    except:
        endo = Polygon()
        epi = Polygon()
    imgsize   = (dcm.Rows, dcm.Columns)
    pixelsize = dcm.PixelSpacing
    anno_dict = dict()
    for cname in ['lv_myo', 'lv_endo', 'lv_epi']:
        anno_dict[cname] = dict()
        anno_dict[cname]['imageSize'] = imgsize
        anno_dict[cname]['pixelsize'] = pixelsize
        anno_dict[cname]['subpixelResolution'] = 1
    anno_dict['lv_myo'] ['cont'] = myo
    anno_dict['lv_myo'] ['contType'] = 'MYO'
    anno_dict['lv_endo']['cont'] = endo
    anno_dict['lv_endo']['contType'] = 'FREE'
    anno_dict['lv_epi'] ['cont'] = epi
    anno_dict['lv_epi'] ['contType'] = 'FREE'
    return anno_dict








def annotation_to_mask(input):
    if type(input) == str and input.endswith(".pickle"):
        file = open(input, "rb")
        anno_dict = pickle.load(file)
        file.close()
    else:
        anno_dict = input

    if "lv_myo" in anno_dict.keys():
        poly = anno_dict["lv_myo"]["cont"]
        mask = features.rasterize([poly], out_shape=anno_dict["lv_myo"]["imageSize"])
    else:
        mask = None
    return mask


def plot_outlines(ax, geo, c=(1,1,1,1.0), shift=0):
    """plots geometry outlines onto matplotlib.pyplot.axis"""
    if geo.geom_type=='Polygon': ax.add_patch(PolygonPatch_Outline(geo, c=c, shift=shift))
    if geo.geom_type=='MultiPolygon':
        for p in geo.geoms:      ax.add_patch(PolygonPatch_Outline(p,   c=c, shift=shift))

def PolygonPatch_Outline(polygon, c=(1,1,1,1.0), alpha=1.0, shift=0):
    return PathPatch(matplotlib.path.Path.make_compound_path(matplotlib.path.Path(np.asarray(polygon.exterior.coords)[:,:2]+shift), *[matplotlib.path.Path(np.asarray(ring.coords)[:,:2]+shift) for ring in polygon.interiors]), ec=c, alpha=alpha, fill=False, lw=0.5)

def PolygonShift(polygon, shift):
    return affinity.affine_transform(polygon, [1, 0, 0, 1, shift, shift])

def ring_mask(h, w, inner_r, outer_r):
    img = np.zeros((h,w), dtype=np.uint8)
    my, mx = h//2, w//2
    for y in range(h):
        for x in range(w):
            if inner_r <= np.sqrt((y-my)**2+(x-mx)**2) <= outer_r:
                img[y,x]=1
    return img


def example():
    mask = ring_mask(200,200, 40,60)
    cont = to_polygon(mask)

    fig, axes = plt.subplots(1,2)
    axes[0].imshow(mask); axes[1].imshow(np.zeros((200,200)))
    for ax in axes: ax.axis('off')
    axes[0].set_title('Mask'); axes[1].set_title('Polygon')
    plot_outlines(axes[1], cont)

    #%%

    # Example store annotation
    dcm = pydicom.dcmread('/Users/thomas/Desktop/CMR/LazyLuna_Data/Data/SGS/Imgs/M2GS09_/series37001-Body/img0002-72.402.dcm')
    sop = dcm.SOPInstanceUID
    anno = mask_to_LL_annotation(mask, dcm)
    with open(os.path.join('/Users/thomas/Desktop', sop+'.pickle'), 'wb') as f: pickle.dump(anno, f)

    print(anno)
    fig, ax = plt.subplots(1,1)
    ax.imshow(np.zeros((200,200))); ax.axis('off')
    ax.set_title('Mask')
    plot_outlines(ax, anno['lv_myo']['cont'])


def old_prepare4LazyLunaSAX(path, data, contours, subpixelresolution=2, reader="me", num_contours=None, save_dicom=True):
    dcm = copy.deepcopy(data)
    dcm[0x0008, 0x103e].value = "OVERWRITTEN: pre_MOLLI MOCO T1 mit"
    h, w = dcm.pixel_array.shape[0], dcm.pixel_array.shape[1]
    res = np.array(dcm.PixelSpacing).astype("float")
    sopuid = dcm.SOPInstanceUID

    contours_new = copy.deepcopy(contours)

    if num_contours is None:
        num_contours = len(contours_new)

    try:
        for i in range(len(contours_new)):
            array = contours_new[i]
            array[:, [0,1]] = array[:, [1,0]]
            contours_new[i] = array
    except:
        pass

    conts_lv1 = [c.tolist() for c in contours_new]

    if len(conts_lv1)>=2 and num_contours==2:
        #conts_lv1=[[[b,a] for c in conts_lv1 for d in c for [a,b] in d]]

        lv_epi = [conts_lv1[-1]]
        lv_endo = [conts_lv1[-2]]

        d = {'lv endocardium':      {'contour': lv_endo,
                                      'type': ['type', '  FREE'],
                                      'subpixel resolution': subpixelresolution,
                                      'width':  w,
                                      'height': h,
                                      'pixel size': res},
              'lv epicardium':       {'contour': lv_epi,
                                      'type': ['type', '  FREE'],
                                      'subpixel resolution': subpixelresolution,
                                      'width':  w,
                                      'height': h,
                                      'pixel size': res},
              'lv papillary muscle': {'contour': []},
              'rv endocardium':      {'contour': []},
              'rv epicardium': {'contour': []},
              'rv papillary muscle': {'contour': []},
              None: {'contour': []},
              'left atrium': {'contour': []},
              'viewer': {'hotspot': [93.7963, 102.087], 'zoom': 4.74074}, # ?
              'window': {'center': 372.0, 'width': 890.0}}                # ?


    elif len(conts_lv1)==1 or num_contours==1:
        lv_epi = [conts_lv1[-1]]

        d = {'lv endocardium':      {'contour': []},
              'lv epicardium':       {'contour': lv_epi,
                                      'type': ['type', '  FREE'],
                                      'subpixel resolution': subpixelresolution,
                                      'width':  w,
                                      'height': h,
                                      'pixel size': res},
              'lv papillary muscle': {'contour': []},
              'rv endocardium':      {'contour': []},
              'rv epicardium': {'contour': []},
              'rv papillary muscle': {'contour': []},
              None: {'contour': []},
              'left atrium': {'contour': []},
              'viewer': {'hotspot': [93.7963, 102.087], 'zoom': 4.74074}, # ?
              'window': {'center': 372.0, 'width': 890.0}}                # ?
    else:
        return False

    if save_dicom:
        try:
            path_dicom = os.path.join(path, "MARISSA2LazyLuna", "images", sopuid)
            os.makedirs(path_dicom)
            dcm.save_as(os.path.join(path_dicom, sopuid+'.dcm'))
        except:
            pass

    try:
        path_annotation = os.path.join(path, "MARISSA2LazyLuna", reader, sopuid, "sub_annotations")
        path_contour_file = os.path.join(path, "MARISSA2LazyLuna", reader, sopuid)
        os.makedirs(path_annotation)
    except:
        pass

    open(os.path.join(path_contour_file, "contours.txt"), "w+").close()
    with open(os.path.join(path_annotation, sopuid+'.pickle'), 'wb') as f:
        pickle.dump(d, f)
        f.close()

    return True


