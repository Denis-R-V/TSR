import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

def draw_bboxes_patches(model, img: str | JpegImageFile,
                        detector_threshold: float = None, classifier_threshold: float = None, debug_mode: float = None):
    
    # Если заданы threshold или debug_mode - меняем параметры модели
    if detector_threshold is not None:
        detector_threshold_old = model.detector_threshold
        model.detector_threshold = detector_threshold
    if classifier_threshold is not None:
        classifier_threshold_old = model.classifier_threshold
        model.classifier_threshold = classifier_threshold
    if debug_mode is not None:
        debug_mode_old = model.debug_mode
        model.debug_mode = debug_mode

    # загрузка изображения, если на вход подан путь
    if isinstance(img, str) == True:
        img = Image.open(img)

    # получение предсказаний модели
    bboxes, labels, detector_scores, classifier_scores = model.predict_single(img)

    # Описание знаков
    signs_in_predict = sorted(list(set([model.class2label_map[label] for label in labels])))
    description_predict = ["{}: {}".format(sign, model.labels2names_map[sign]) for sign in signs_in_predict]
    print('\n'.join(description_predict))

    # plot the image and bboxes
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(18,10)
    a.imshow(img) 

    # Bounding boxes are defined as follows: x-min y-min width height
    for i in range(len(bboxes)):
        x, y, width, height  = bboxes[i][0], bboxes[i][1], bboxes[i][2]-bboxes[i][0], bboxes[i][3]-bboxes[i][1]
        rect = patches.Rectangle((x, y), width, height, linewidth = 2, edgecolor = 'yellow', facecolor = 'none')
        # Draw the bounding box on top of the image
        a.add_patch(rect)

    for i in range(len(bboxes)):
        text_x = round(bboxes[i][0])
        text_y = round(bboxes[i][1])
        label = "{0}: {1:1.2f}/{2:1.2f}".format(str(model.class2label_map[labels[i]]), detector_scores[i], classifier_scores[i])
        plt.text(text_x-10 , text_y-10, label, fontdict = {'family': 'arial', 'color': 'red', 'weight': 'bold', 'size': 10})

    plt.show()

    # Возврат threshold или debug_mode
    if detector_threshold is not None: model.detector_threshold = detector_threshold_old
    if classifier_threshold is not None: model.classifier_threshold = classifier_threshold_old
    if debug_mode is not None: model.debug_mode = debug_mode_old

    return bboxes, labels, detector_scores, classifier_scores