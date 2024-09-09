import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw, ImageFont
from PIL.JpegImagePlugin import JpegImageFile
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

class Builder:
    def __init__(self,
                 device: str = 'cpu',
                 class2label_path: str = os.path.join('data', 'prepared', 'label_map.json'),
                 label2name_path: str = os.path.join('data', 'prepared', 'labels_names_map.json'),
                 detector_path: str = os.path.join('models', 'chkpt_detector_resnet50_v2_augmented_b8_5.pth'),
                 classifier_path: str = os.path.join('models', 'classifier_resnet152_add_signs_bg100_tvs_randomchoice_perspective_colorjitter_resizedcrop_erasing_adam_001_sh_10_06_model_29.pth'),
                 detector_num_classes: int = 2,
                 classifier_num_classes: int = 156,
                 detector_threshold: float = 0.,
                 classifier_threshold: int = 0.,
                 debug_mode: bool = False):
        """Загрузка ML модели и вспомогательных файлов"""
        
        self.device = device
        self.detector_threshold = detector_threshold
        self.classifier_threshold = classifier_threshold
        self.debug_mode = debug_mode

        self.__load_class2label_map(class2label_path)
        self.__load_label2name_map(label2name_path)
        self.__load_detector(detector_path, detector_num_classes)
        self.__load_classisier(classifier_path, classifier_num_classes)

    def __load_class2label_map(self, path):
        """Загрузка маппинга ID классов на ID знаков"""
 
        with open(path, 'r') as read_file:
            self.class2label_map = json.load(read_file)

        self.class2label_map = {v:k for k, v in self.class2label_map.items()}
        self.class2label_map = {0: 'bg', **self.class2label_map}

    def __load_label2name_map(self, path):
        """Загрузка маппинга ID знаков на их название"""
      
        with open(path, 'r') as read_file:
            self.labels2names_map = json.load(read_file)
        read_file.close()
        self.labels2names_map = {'bg': 'Background (ложная детекция)', **self.labels2names_map}
    
    def __load_detector(self, path, num_classes):
        """Загрузка детектора"""

        # Добавить detector_path для работы только детектора или классификатора
        
        # Загрузка модели
        loaded_model = torch.load(path, map_location=self.device, weights_only=False)

        # Если загружен файл модели
        if type(loaded_model).__name__ != 'dict':
            self.detector = loaded_model
            self.detector.to(self.device)
            self.detector.eval()

            print(f'Загружен детектор FasterRCNN из {path}')

        # Если загружен чекпойнт с весами
        else:
            if ('resnet50_v2' in path) or ('resnet50v2' in path):
                backbone = 'resnet50v2'
                self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
            elif 'resnet50' in path:
                backbone = 'resnet50'
                self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            elif 'mobilenet_v3_320' in path:
                backbone = 'mobilenet_v3_320'
                self.detector =torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
            elif 'mobilenet_v3' in path:
                backbone = 'mobilenet_v3'
                self.detector =torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
            else:
                self.detector = None

            if self.detector is None:
                print("Неверно указан путь к детектору")
            else:
                
                # get number of input features for the classifier
                in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
                # replace the pre-trained head with a new one
                self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                
                # Загрузка весов модели
                self.detector.load_state_dict(loaded_model['model_state_dict'])
                self.detector.to(self.device)
                self.detector.eval()

                print(f'Для FasterRCNN с backbone {backbone} загружены веса из {path}')

    def __load_classisier(self, path, num_classes):
        """Загрузка классификатора"""
        
        # Загрузка модели
        loaded_model = torch.load(path, map_location=self.device, weights_only=False)

        # Если загружен файл модели
        if type(loaded_model).__name__ != 'dict':
            self.classifier = loaded_model
            self.classifier.to(self.device)
            self.classifier.eval()

            print(f'Загружен классификатор из {path}')
        
        # Если загружен чекпойнт с весами
        else:        
            if 'resnet152' in path:
                classifier_name = 'ResNet-152'
                self.classifier = torchvision.models.resnet152(weights=None)
                for param in self.classifier.parameters():
                    param.requires_grad = False       

                self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
                for param in self.classifier.fc.parameters():
                    param.requires_grad = True

            else:
                self.classifier = None
                
            if self.classifier is None:
                print("Неверно указан путь к классификатору")
                
            else:            
                # Загрузка весов модели
                self.classifier.load_state_dict(loaded_model['model_state_dict'])
                self.classifier.to(self.device)
                self.classifier.eval()

                print(f"Для {classifier_name} загружены веса {path}")

    def preprocessing_single(self, payload: str | np.ndarray | JpegImageFile):
        """Препроцессинг для онлайн предсказания"""
        
        if isinstance(payload, str) == True:
            img = Image.open(payload)
        elif isinstance(payload, np.ndarray):
            img = cv2.cvtColor(payload, cv2.COLOR_BGR2RGB)
        else:
            img = payload

        return img

    def predict_signs(self, detector_input):
        """Детекция знаков на изображении"""
        
        #detector_input = v2.functional.to_tensor(detector_input).to(self.device)
        # The function `to_tensor(...)` is deprecated and will be removed in a future release.
        # Instead, please use `to_image(...)` followed by `to_dtype(..., dtype=torch.float32, scale=True)
        detector_input = v2.functional.to_image(detector_input)
        detector_input = v2.functional.to_dtype(detector_input, dtype=torch.float32, scale=True).to(self.device)

        with torch.no_grad():
            detector_pred = self.detector([detector_input])

        bboxes = [[i[0], i[1], i[2], i[3]] for i in list(detector_pred[0]['boxes'].detach().cpu().numpy())]
        pred_detector_labels = list(detector_pred[0].get('labels').detach().cpu().numpy())
        pred_detector_scores = list(detector_pred[0].get('scores').detach().cpu().numpy())

        # Фильтрация предсказаний с уверенностью модели больше threshold
        if (self.debug_mode == False) and (self.detector_threshold > 0.):       
            try:
                pred_tr = [pred_detector_scores.index(x) for x in pred_detector_scores if x > self.detector_threshold][-1]
                bboxes = bboxes[:pred_tr+1]
                pred_detector_labels = pred_detector_labels[:pred_tr+1] 
                pred_detector_scores = pred_detector_scores[:pred_tr+1]
            except:
                bboxes = []
                pred_detector_labels = []
                pred_detector_scores = []

        return bboxes, pred_detector_labels, pred_detector_scores

    def predict_class(self, img):
        """Классификация знака"""

        with torch.no_grad():
            pred_classifier = self.classifier(img.unsqueeze(0))
            pred_classifier = torch.nn.functional.softmax(pred_classifier, dim=1).cpu().max(1,keepdim=True)

        if (self.debug_mode == False) and (float(pred_classifier.values[0][0]) < self.classifier_threshold):
            pred_classifier.indices[0][0] = 0

        return pred_classifier

    def predict_single(self, model_input: str | np.ndarray | JpegImageFile):
        """Предсказание онлайн"""
        '''
        на вход подается путь к изображению или изображение, открытое PIL или OpenCV (BGR) и пороги чувствительности детектора и классификатора
        функция возвращает координаты рамок, ID классов, уверенности детекций и вероятности класса для одного изображения
        при debug_mode=True выводятся детекции с любой уверенностью и 0-й класс (фон)
        '''

        assert isinstance(model_input, str) or isinstance(model_input, np.ndarray) or isinstance(model_input, JpegImageFile), \
            'На вход модели подается путь к изображению или изображение, открытое PIL или OpenCV (BGR)'

        img = self.preprocessing_single(model_input)
        bboxes, pred_labels, pred_detector_scores = self.predict_signs(img)

        pred_labels = []
        pred_classifier_scores = []
        for i in range(len(bboxes)):
            if img.__class__.__name__ == 'ndarray':
                sign = img[round(bboxes[i][1]):round(bboxes[i][3]), round(bboxes[i][0]):round(bboxes[i][2])]
            else:
                sign = img.crop(bboxes[i])
  
            #sign = v2.functional.to_tensor(sign)
            # The function `to_tensor(...)` is deprecated and will be removed in a future release.
            # Instead, please use `to_image(...)` followed by `to_dtype(..., dtype=torch.float32, scale=True)
            sign = v2.functional.to_image(sign)
            sign = v2.functional.to_dtype(sign, dtype=torch.float32, scale=True)
            sign = v2.functional.resize(sign, [224,224]).to(self.device)       
            pred_classifier = self.predict_class(sign)
            
            pred_labels.append(int(pred_classifier.indices[0][0]))
            pred_classifier_scores.append(float(pred_classifier.values[0][0]))

        # Если режим отладки не включен убираем детекции, которые классифицированы как фон
        if self.debug_mode == False:
            # индексы 0 класса (фона) в предсказаниях классификатора
            background_indexes = [index for index, label in enumerate(pred_labels) if label ==0]
            if background_indexes != []:
                background_indexes.reverse()
                # удаляем предикты для всех результатов для 0 класса (фона)
                for index in background_indexes:
                    bboxes.pop(index)
                    pred_labels.pop(index)
                    pred_detector_scores.pop(index)
                    pred_classifier_scores.pop(index)
        
        return bboxes, pred_labels, pred_detector_scores, pred_classifier_scores
    
    def __get_font_path(self, font_name):
        """Поиск пути к шрифту"""
    
        paths = [os.path.join('..', 'data', 'prepared', font_name),
                os.path.join('..', 'data', font_name),
                os.path.join('data', 'prepared', font_name),
                os.path.join('data', font_name),
                os.path.join('C:\\Windows\\Fonts', font_name),
                os.path.join('/System/Library/Fonts/Supplemental', font_name),
                os.path.join('/System/Library/Fonts', font_name)]
        
        font_path = None
        for path in paths:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path is None:
            print(f'Шрифт {font_name} не найден')

        return font_path
    
    def __draw_bboxes_pil(self, img, bboxes, labels, detector_scores, classifier_scores, display_img, save_path):
        """Добавление рамок и описаний на изображение, открытое PIL"""
 
        rectangle_thickness = round(img.width/500)
        text_size = round(img.height/60)+4      #round(img.height/50)
        font = 'Arial Black.TTF'                #'ARLRDBD.TTF'
        font = ImageFont.truetype(self.__get_font_path(font), size=text_size)     

        # ImageDraw  отрисовывает рамки и трешхолды непосредственно на изображении
        pencil = ImageDraw.Draw(img)
        for i in range(len(bboxes)):
            #pencil.rectangle(bboxes[i], fill = None, width=8, outline='yellow')
            pencil.rectangle(bboxes[i], fill = None, width=rectangle_thickness, outline='yellow')
        for i in range(len(bboxes)):
            text_x = round(bboxes[i][0]) - 10
            text_y = round(bboxes[i][1]) - text_size - rectangle_thickness/2
            label = "{0}: {1:1.2f}/{2:1.2f}".format(str(self.class2label_map[labels[i]]), detector_scores[i], classifier_scores[i])
            pencil.text((text_x, text_y), label, font=font, fill = 'red')  
        
        if save_path:
            img.save(save_path)

        if display_img:
            fig, a = plt.subplots()
            fig.set_size_inches(18, 10)
            a.imshow(img)
            plt.show()

        return img

    def __draw_bboxes_opencv(self, img, bboxes, labels, detector_scores, classifier_scores, display_img, save_path):
        """Добавление рамок и описаний на изображение, открытое OpenCV"""
        # font  FONT_HERSHEY_SIMPLEX FONT_HERSHEY_PLAIN FONT_HERSHEY_DUPLEX FONT_HERSHEY_COMPLEX FONT_HERSHEY_TRIPLEX
        #       FONT_HERSHEY_COMPLEX_SMALL FONT_HERSHEY_SCRIPT_SIMPLEX FONT_HERSHEY_SCRIPT_COMPLEX
        font = cv2.FONT_HERSHEY_COMPLEX
        
        # fontScale
        fontScale = 1.5
        
        # Colors in BGR
        rect_color = (0, 255, 255)
        text_color = (0, 0, 255)

        # Line thickness of px
        thickness = 4

        for i in range(len(bboxes)):
            cv2.rectangle(img, (round(bboxes[i][0]), round(bboxes[i][1])), (round(bboxes[i][2]), round(bboxes[i][3])), rect_color, 8)
        for i in range(len(bboxes)):    
            text_x = round(bboxes[i][0]) - 10
            text_y = round(bboxes[i][1]) - 10  
            mark = "{0}: {1:1.2f}/{2:1.2f}".format(str(self.class2label_map[labels[i]]), detector_scores[i], classifier_scores[i])
            img = cv2.putText(img, mark, (text_x, text_y), font, 
                    fontScale, text_color, thickness, cv2.LINE_AA, False)

        if save_path:
            cv2.imwrite(save_path, img)

        if display_img:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig, a = plt.subplots(1,1)
            fig.set_size_inches(18,10)
            a.imshow(img_rgb)

        return img


    def predict_single_visualized(self, img: str | np.ndarray | JpegImageFile, display_img: bool = False, save_path: str = None,
                                  detector_threshold: float = None, classifier_threshold: float = None, debug_mode: float = None):
        """Предикт с возвратом изображения с рамками и описаниями
        на вход подается путь к изображению или изображение, открытое PIL или OpenCV (BGR) и пороги чувствительности детектора и классификатора
        на выходе изображение PIL с нанесенными рамками, ID классов, уверенностями детекций и вероятностями класса для одного изображения  
        (OpenCV в BGR, если на вход подавальсь изображение, открытое OpenCV в BGR) и список с расшифровкой классов
        при display_img=False
        при debug_mode=True выводятся детекции с любой уверенностью и  0-й класс (фон)
        пороги чувствительности детектора и классификатора меняются только для одного предсказания
        """

        # Если заданы threshold или debug_mode - меняем параметры модели
        if detector_threshold is not None:                                  # перенести в предикт
            detector_threshold_old = self.detector_threshold
            self.detector_threshold = detector_threshold
        if classifier_threshold is not None:
            classifier_threshold_old = self.classifier_threshold
            self.classifier_threshold = classifier_threshold
        if debug_mode is not None:
            debug_mode_old = self.debug_mode
            self.debug_mode = debug_mode

        # загрузка изображения, если на вход подан путь
        if isinstance(img, str) == True:
            img = Image.open(img)

        # получение предсказаний модели
        bboxes, labels, detector_scores, classifier_scores = self.predict_single(img)

        # Описание знаков
        signs_in_predict = sorted(list(set([self.class2label_map[label] for label in labels])))
        description_predict = ["{}: {}".format(sign, self.labels2names_map[sign]) for sign in signs_in_predict]
        if display_img:
            print('\n'.join(description_predict))

        # если изображение открыто PIL
        if isinstance(img, JpegImageFile) == True:
            img_pred = self.__draw_bboxes_pil(img, bboxes, labels, detector_scores, classifier_scores, display_img, save_path)
        # если изображение открыто OpenCV
        else:
            img_pred = self.__draw_bboxes_opencv(img, bboxes, labels, detector_scores, classifier_scores, display_img, save_path)

        # Возврат threshold или debug_mode
        if detector_threshold is not None: self.detector_threshold = detector_threshold_old
        if classifier_threshold is not None: self.classifier_threshold = classifier_threshold_old
        if debug_mode is not None: self.debug_mode = debug_mode_old

        return img_pred, description_predict