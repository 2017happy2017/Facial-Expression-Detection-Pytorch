"""
Image processing utils

Inspired by many reference code from tensorflow and keras.

Zuzeng Lin, 2018
"""
import cv2
import numpy as np
import torchvision.transforms as transforms


def resize_image(img, scale):
    """
    Args:
        img: input image, numpy array , height x width x channel
        scale: scale factor
    Rets:
        resized image, tensor , 1 x channel x height x width

    A simple cv2.resize warpper using a scaling factor.
    """
    height, width, channels = img.shape
    new_dim = (int(width * scale), int(height * scale))
    return cv2.resize(
        img, new_dim, interpolation=cv2.INTER_LINEAR)


def convert_to_square(bbox):
    """
    Args:
        bbox: bounding box, numpy array, see detect.py for details
    Rets:
        square bounding box, numpy array 

    Convert arbitrary bounding boxes to square boxes.
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    long_side = np.maximum(h, w)  # take the longer side as the new length
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - long_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - long_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + long_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + long_side - 1
    return square_bbox


def IoU(box, boxes):
    """
    Args:
        box: bounding box to be tested, numpy array, see detect.py for details
        boxes: ground truth bounding boxes, numpy array, shape (n, 4)

    Rets:
        IoU values, numpy array

    Evaluate intersection over union ratio.
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx_upleft = np.maximum(box[0], boxes[:, 0])
    yy_upleft = np.maximum(box[1], boxes[:, 1])
    xx_downright = np.minimum(box[2], boxes[:, 2])
    yy_downright = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx_downright - xx_upleft + 1)
    h = np.maximum(0, yy_downright - yy_upleft + 1)
    overlap = w * h
    iou = np.true_divide(overlap, (box_area + area - overlap))
    return iou


def neighbour_supression(bboxes, thresh, mode="Union"):
    """
    Args:
        bboxes: bounding boxes, numpy array, n x 5
        thresh: thershold
        mode: Union / Minimum
    Rets:
        indexes in bboxes to keep

    Select bounding boxes with high confidence, and overlap <= thresh. 

    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        # select largest
        i = sorted_idx[0]
        keep.append(i)
        # calculate intersection area
        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        if mode == "Union":
            ovr = intersection / \
                (areas[i] + areas[sorted_idx[1:]] - intersection)
        elif mode == "Minimum":
            ovr = intersection / np.minimum(areas[i], areas[sorted_idx[1:]])
        # keep low overlapping results for next iteration
        idx = np.where(ovr <= thresh)[0]
        sorted_idx = sorted_idx[idx + 1]

    return keep


def bbox_crop(img, bboxes, resized=(48, 48), totensor=True):
    """
    Args:
        img: source image
        bboxes: bounding boxes for crop, numpy array, n x 5
        resized: size of output crops
        totensor: format of optput crops
    Rets:
        list of numpy array or tensor, crop images

    Crop images using bounding boxes.
    """

    h, w, c = img.shape
    # convert to integers
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
    w_tmp = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
    h_tmp = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)

    # create slices storage
    numbox = bboxes.shape[0]
    dx = np.zeros((numbox, ))
    dy = np.zeros((numbox, ))
    edx, edy = w_tmp.copy()-1, h_tmp.copy()-1

    # get slices from bboxes
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # deal with out of screen cases
    idx_tmp = np.where(ex > w-1)
    edx[idx_tmp] = w_tmp[idx_tmp] + w - 2 - ex[idx_tmp]
    ex[idx_tmp] = w - 1

    idx_tmp = np.where(ey > h-1)
    edy[idx_tmp] = h_tmp[idx_tmp] + h - 2 - ey[idx_tmp]
    ey[idx_tmp] = h - 1

    idx_tmp = np.where(x < 0)
    dx[idx_tmp] = 0 - x[idx_tmp]
    x[idx_tmp] = 0

    idx_tmp = np.where(y < 0)
    dy[idx_tmp] = 0 - y[idx_tmp]
    y[idx_tmp] = 0

    # convert to integers
    [dy, edy, dx, edx, y, ey, x, ex, w_tmp, h_tmp] = [item.astype(
        np.int32) for item in [dy, edy, dx, edx, y, ey, x, ex, w_tmp, h_tmp]]

    # crop image using prepared slices
    cropped_tmp_tensors = []
    for i in range(bboxes.shape[0]):
        dest_img_tmp = np.zeros((h_tmp[i], w_tmp[i], 3), dtype=np.uint8)
        dest_img_tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1,
                     :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        resized_tmp = cv2.resize(dest_img_tmp, resized)
        if totensor:
            cropped_tmp_tensors.append(transforms.ToTensor()(resized_tmp))
        else:
            cropped_tmp_tensors.append(resized_tmp)
    return cropped_tmp_tensors
