"""
Graphical User Interface

Zuzeng Lin, 2018
"""
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from detect import CNNDetector


def pltshow():
    plt.gcf().canvas.set_window_title(
        "Facial Expression Detection")
    figman = plt.get_current_fig_manager()
    figman.window.state('zoomed')
    plt.show()


def read_folder(path):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
            for name in files
            ]


if __name__ == "__main__":

    useage = """
    Facial Expression Detection

    Useage:

    This program assumes that there is **AT LEAST ONE** face in each image.

    Put the all the face images (jpeg files) in the ./images/ folder,
    and make sure the ./parameters/ folder cotains trained weights files.

    Please note that rotated faces are NOT supported.
    Faces shot from the side or with glasses might not be recognized correctly.
    Images with more than one face will slow down the program.
    
    """
    print(useage)

    cnn_detector = CNNDetector(net_12_param_path="./parameters/12_net.pt",
                               net_48_param_path="./parameters/48_net.pt",
                               net_vgg_param_path="./parameters/vgg_net.pt",
                               )
    image_list = read_folder("images/")
    fig = plt.figure()
    plt.text(0, 0, (useage), ha='left', rotation=0,  va='top', wrap=True)
    plt.ylim(-5, 0)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    pltshow()
    for filename in image_list:
        print(filename)
        try:
            img = cv2.imread(filename)
            bboxes, _ = cnn_detector.detect_face(img)
            # display face detection results
            fig, ax = plt.subplots(1)

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.suptitle('Face Detection - %s' % filename, fontsize=15)
            # draw bounding box
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]),
                                         bbox[2] - bbox[0],
                                         bbox[3] - bbox[1],
                                         facecolor='none', edgecolor='w',
                                         linewidth=2)
                ax.add_patch(rect)
            pltshow()

            face_imgs = cnn_detector.crop_faces(img, bboxes)
            for each in face_imgs:
                probabilities, predicted_class = cnn_detector.vgg_net(each)

                # display image
                ax = plt.subplot(1, 2, 1)
                emotion_names = ['Angry', 'Disgust', 'Fear',
                                 'Happy', 'Sad', 'Surprise', 'Neutral']
                plt.xlabel('Face - %s' %
                           str(emotion_names[predicted_class]), fontsize=25)
                plt.imshow(cv2.cvtColor(each, cv2.COLOR_BGR2RGB))
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                # display bar chart
                plt.subplot(1, 2, 2)
                plt.ylabel("Probability", fontsize=15)
                indexes = np.arange(len(emotion_names))

                barlist = plt.bar(indexes, probabilities,
                                  align='center', alpha=0.5)
                barlist[predicted_class].set_color('r')
                plt.xticks(indexes, emotion_names, rotation=30, fontsize=15)
                plt.tight_layout()
                pltshow()
        except Exception as ex:
            print(ex)
