"""
Facical Expression Detector Classes

Zuzeng Lin, 2018
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd.variable import Variable

import imageproc 
from models.models import Net12, Net48
from models.vgg import VGG


class CNNDetector(object):
    def __init__(self, net_12_param_path=None, net_48_param_path=None, net_vgg_param_path=None, 
    use_cuda=True, pthreshold=0.7,rthershold=0.9):
        if use_cuda == False:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        if net_12_param_path is not None:
            self.net_12 = Net12()
            self.net_12.load_state_dict(torch.load(
                net_12_param_path, map_location=lambda storage, loc: storage))
            self.net_12.to(self.device)
            self.net_12.eval()
        if net_48_param_path is not None:
            self.net_48 = Net48()
            self.net_48.load_state_dict(torch.load(
                net_48_param_path, map_location=lambda storage, loc: storage))
            self.net_48.to(self.device)
            self.net_48.eval()
        if net_vgg_param_path is not None:
            self.net_vgg = VGG('VGG19')
            self.net_vgg.load_state_dict(torch.load(
                net_vgg_param_path, map_location=lambda storage, loc: storage))
            self.net_vgg.to(self.device)
            self.net_vgg.eval()
        self.pthreshold = pthreshold
        self.rthershold = rthershold 

    def generate_stage(self, img):
        """
        Args:
            img: source image
        Rets:
            bounding boxes, numpy array, n x 5

        Generate face bounding box proposals using net-12.
        """
        proposals = list()
        downscaling_factor = 0.7
        current_height, current_width, _ = img.shape
        current_scale = 1.0
        # limit maximum height to 500
        if current_height>500:
            current_scale = 500.0/current_height

        receptive_field = 12
        stride = 2
        while True:
            # get the resized image at current scale
            im_resized = imageproc.resize_image(img, current_scale)
            current_height, current_width, _ = im_resized.shape
            if min(current_height, current_width) <= receptive_field:  # receptive field of the net-12
                break
            # transpose hwc (Numpy) to chw (Tensor)
            feed_imgs = (transforms.ToTensor()(
                im_resized)).unsqueeze(0).float()
            # feed to net-12
            with torch.no_grad():
                feed_imgs = feed_imgs.to(self.device)
                bbox_class, bbox_regress = self.net_12(feed_imgs)
              
                bbox_class = bbox_class.cpu().squeeze(0).detach().numpy()
                bbox_regress = bbox_regress.cpu().squeeze(0).detach().numpy()

            # FILTER classes with threshold
            up_thresh_masked_index = np.where(
                bbox_class > self.pthreshold)  # threshold
            up_thresh_masked_index = up_thresh_masked_index[1:3]
            filtered_results = np.vstack([
                # pixel coordinate for receptive window
                np.round((stride * up_thresh_masked_index[1]) / current_scale),
                np.round((stride * up_thresh_masked_index[0]) / current_scale),
                np.round(
                    (stride * up_thresh_masked_index[1] + receptive_field) / current_scale),
                np.round(
                    (stride * up_thresh_masked_index[0] + receptive_field) / current_scale),
                # original bbox output form network
                bbox_class[0, up_thresh_masked_index[0],
                           up_thresh_masked_index[1]],
                bbox_regress[:, up_thresh_masked_index[0],
                             up_thresh_masked_index[1]],
            ]).T
            keep_mask = imageproc.neighbour_supression(filtered_results[:, :5], 0.7, 'Union')
            filtered_results = filtered_results[keep_mask]
            current_scale *= downscaling_factor
            proposals.append(filtered_results)
        # aggregate proposals from list
        proposals = np.vstack(proposals)
        keep_mask = imageproc.neighbour_supression(proposals[:, 0:5], 0.5, 'Union')
        proposals = proposals[keep_mask]
        if len(proposals) == 0:
            # no proposal generated
            return None
        # convert multi-sacle bbox to unified bbox at original img scale
        receptive_window_width_pixels = proposals[:, 2] - proposals[:, 0] + 1
        receptive_window_height_pixels = proposals[:, 3] - proposals[:, 1] + 1
        bbox_aligned = np.vstack([
            proposals[:, 0] + proposals[:, 5] *
            receptive_window_width_pixels,  # upleft_x
            proposals[:, 1] + proposals[:, 6] * \
            receptive_window_height_pixels,  # upleft_y
            proposals[:, 2] + proposals[:, 7] * \
            receptive_window_width_pixels,  # downright_x
            proposals[:, 3] + proposals[:, 8] * \
            receptive_window_height_pixels,  # downright_y
            proposals[:, 4],  # classes
        ])
        bbox_aligned = bbox_aligned.T

        return bbox_aligned

    def refine_stage(self, img, proposal_bbox):
        """
        Args:
            img: source image
            proposal_bbox: bounding box proposals from generate stage 
        Rets:
            bounding boxes, numpy array, n x 5

        Apply delta corrdinate to bboxes using net-48.
        """
        if proposal_bbox is None:
            return None, None

        proposal_bbox = imageproc.convert_to_square(proposal_bbox)

        cropped_tmp_tensors = imageproc.bbox_crop(img, proposal_bbox)
        # feed to net-48
        with torch.no_grad():
            feed_imgs = Variable(torch.stack(cropped_tmp_tensors))

            feed_imgs = feed_imgs.to(self.device)
        
            bbox_class, bbox_regress, landmark = self.net_48(feed_imgs)

            bbox_class = bbox_class.cpu().detach().numpy()
            bbox_regress = bbox_regress.cpu().detach().numpy()
            landmark = landmark.cpu().detach().numpy()
        # threshold
        up_thresh_masked_index = np.where(bbox_class > self.rthershold)[0]
        boxes = proposal_bbox[up_thresh_masked_index]
        bbox_class = bbox_class[up_thresh_masked_index]
        bbox_regress = bbox_regress[up_thresh_masked_index]
        landmark = landmark[up_thresh_masked_index]
        # aggregate
        keep_mask = imageproc.neighbour_supression(boxes, 0.5, mode="Minimum")

        if len(keep_mask) == 0:
            return None, None

        proposals = boxes[keep_mask]
        bbox_class = bbox_class[keep_mask]
        bbox_regress = bbox_regress[keep_mask]
        landmark = landmark[keep_mask]

        receptive_window_width_pixels = proposals[:, 2] - proposals[:, 0] + 1
        receptive_window_height_pixels = proposals[:, 3] - proposals[:, 1] + 1
        # get new bounding boxes
        boxes_align = np.vstack([
            proposals[:, 0] + bbox_regress[:, 0] *
            receptive_window_width_pixels,  # upleft_x
            proposals[:, 1] + bbox_regress[:, 1] *
            receptive_window_height_pixels,  # upleft_y
            proposals[:, 2] + bbox_regress[:, 2] *
            receptive_window_width_pixels,  # downright_x
            proposals[:, 3] + bbox_regress[:, 3] *
            receptive_window_height_pixels,  # downright_y
            bbox_class[:, 0],
        ]).T
        # get facial landmarks
        align_landmark_topx = proposals[:, 0]
        align_landmark_topy = proposals[:, 1]
        landmark_align = np.vstack([
            align_landmark_topx + landmark[:, 0] *
            receptive_window_width_pixels,  # lefteye_x
            align_landmark_topy + landmark[:, 1] *
            receptive_window_height_pixels,  # lefteye_y
            align_landmark_topx + landmark[:, 2] *
            receptive_window_width_pixels,  # righteye_x
            align_landmark_topy + landmark[:, 3] *
            receptive_window_height_pixels,  # righteye_y
            align_landmark_topx + landmark[:, 4] *
            receptive_window_width_pixels,  # nose_x
            align_landmark_topy + landmark[:, 5] *
            receptive_window_height_pixels,  # nose_y
            align_landmark_topx + landmark[:, 6] *
            receptive_window_width_pixels,  # leftmouth_x
            align_landmark_topy + landmark[:, 7] *
            receptive_window_height_pixels,  # leftmouth_y
            align_landmark_topx + landmark[:, 8] *
            receptive_window_width_pixels,  # rightmouth_x
            align_landmark_topy + landmark[:, 9] *
            receptive_window_height_pixels,  # rightmouth_y
        ]).T

        return boxes_align, landmark_align

    def detect_face(self, img, atleastone=True):
        """
        Args:
            img: source image
            atleastone: whether the size of image should be retured when no face is found
        Rets:
            bounding boxes, numpy array
            landmark, numpy array

        Detect faces in the image. 
        """
        if self.net_12:
            boxes_align = self.generate_stage(img)
        if self.net_48:
            boxes_align, landmark_align = self.refine_stage(img, boxes_align)
        if boxes_align is None:
            if atleastone:
                boxes_align = np.array([[0, 0, img.shape[1], img.shape[0]]])
            else:
                boxes_align = np.array([])
        if landmark_align is None:
            landmark_align = np.array([])
        return boxes_align, landmark_align

    def crop_faces(self, img, bbox=None):
        """
        see imageproc.bbox_crop
        """
        return imageproc.bbox_crop(img, bbox, totensor=False)

    def vgg_net(self, img):
        """
        Args:
            img: source image
        Rets:
            prob of each expression: in order of 
            ['Angry', 'Disgust', 'Fear',
                       'Happy', 'Sad', 'Surprise', 'Neutral'] 

        Detect facial expression in the image. 
        """

        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_img = cv2.resize(grey_img, (48, 48)).astype(np.uint8)

        grey_img = grey_img[:, :, np.newaxis]
        grey_img = np.concatenate((grey_img, grey_img, grey_img), axis=2)
        receptive_field = 44
        # get ten crops at the corners and center
        tencrops = transforms.Compose([
            transforms.ToPILImage(),
            transforms.TenCrop(receptive_field),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
        ])

        inputs = tencrops(grey_img)

        ncrops, c, h, w = np.shape(inputs)
        # feed to VGG net
        with torch.no_grad():
            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.to(self.device)
            outputs = self.net_vgg(inputs)
            # get mean value across all the crops
            outputs_avg = outputs.view(ncrops, -1).mean(0)
            probabilities = F.softmax(outputs_avg, dim=0)
            # max prob as the detection resutlt
            _, predicted_class = torch.max(outputs_avg.data, 0)
            probabilities=probabilities.cpu().numpy()
            predicted_class=int(predicted_class.cpu().numpy())
        return  probabilities, predicted_class
