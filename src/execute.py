import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

class Builder:
    def __init__(self,
                 device:str = 'cpu',
                 detector_path: str = os.path.join('..', 'models', 'chkpt_detector_resnet50_v2_augmented_b8_5.pth'),
                 classifier_path: str = os.path.join('..', 'models', 'classifier_resnet152_add_signs_bg100_tvs_randomchoice_perspective_colorjitter_resizedcrop_erasing_adam_001_sh_10_06_chkpt_29.pth'),
                 detector_num_classes: int = 2,
                 classifier_num_classes: int = 156,
                 detector_threshold: float = 0.,
                 classifier_threshold: int = 0.,
                 debug_mode: bool = False):
        """Загрузка ML модели и вспомогательных файлов"""

        print('Добавить файлы со знаками')
        
        self.device = device
        self.__load_detector(detector_path, detector_num_classes)
        self.__load_classisier(classifier_path, classifier_num_classes)
    
        self.detector_threshold = detector_threshold
        self.classifier_threshold = classifier_threshold
        self.debug_mode = debug_mode
        
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
        """Препроцессинг для онлайн процесса"""
        
        if isinstance(payload, str) == True:
            img = Image.open(payload)
        elif isinstance(payload, np.ndarray):
            img = cv2.cvtColor(payload, cv2.COLOR_BGR2RGB)
        else:
            img = payload

        return img

    def predict_signs(self, detector_input):
        """Детекция знаков на изображении"""
        
        transforms_img=torchvision.transforms.ToTensor()
        detector_input = transforms_img(detector_input).to(self.device)
        
        with torch.no_grad():
            detector_pred = self.detector([detector_input])

        pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(detector_pred[0]['boxes'].detach().cpu().numpy())]
        pred_detector_labels = list(detector_pred[0].get('labels').detach().cpu().numpy())
        pred_detector_scores = list(detector_pred[0].get('scores').detach().cpu().numpy())

        # Фильтрация предсказаний с уверенностью модели больше threshold
        if (self.debug_mode == False) and (self.detector_threshold > 0.):       
            try:
                pred_tr = [pred_detector_scores.index(x) for x in pred_detector_scores if x > self.detector_threshold][-1]
                pred_boxes = pred_boxes[:pred_tr+1]
                pred_detector_labels = pred_detector_labels[:pred_tr+1] 
                pred_detector_scores = pred_detector_scores[:pred_tr+1]
            except:
                pred_boxes = []
                pred_detector_labels = []
                pred_detector_scores = []

        return pred_boxes, pred_detector_labels, pred_detector_scores

    def predict_class(self, img):
        """Классификация знака"""

        with torch.no_grad():
            pred_classifier = self.classifier(img.unsqueeze(0))
            pred_classifier = torch.nn.functional.softmax(pred_classifier, dim=1).cpu().max(1,keepdim=True)

        if (self.debug_mode == False) and (float(pred_classifier.values[0][0]) < self.classifier_threshold):
            pred_classifier.indices[0][0] = 0

        return pred_classifier

    def predict_single(self, model_input: str | np.ndarray | JpegImageFile):
        """Скоринг для онлайн процесса"""
        '''
        на вход подается путь к изображению или изображение, открытое PIL или OpenCV (BGR) и пороги чувствительности детектора и классификатора
        функция возвращает координаты рамок, ID классов, уверенности детекций и вероятности класса для одного изображения
        при debug_mode=True выводятся детекции с любой уверенностью и 0-й класс (фон)
        ''' 

        img = self.preprocessing_single(model_input)
        pred_boxes, pred_labels, pred_detector_scores = self.predict_signs(img)

        pred_labels = []
        pred_classifier_scores = []
        for i in range(len(pred_boxes)):
            if img.__class__.__name__ == 'ndarray':
                sign = img[round(pred_boxes[i][1]):round(pred_boxes[i][3]), round(pred_boxes[i][0]):round(pred_boxes[i][2])]
            else:
                sign = img.crop(pred_boxes[i])
            
            sign = v2.functional.to_tensor(sign)
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
                    pred_boxes.pop(index)
                    pred_labels.pop(index)
                    pred_detector_scores.pop(index)
                    pred_classifier_scores.pop(index)
        
        return pred_boxes, pred_labels, pred_detector_scores, pred_classifier_scores