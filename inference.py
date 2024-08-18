# Система распознавания дорожных знаков на датасете RTSD
# Инференс (Telegram Bot)

# Запуск в colab или локально
# если работаем в колабе - монтируем диск
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    colab=True
except:
    colab=False

import json
import os
import re

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import telebot

from PIL import Image
from PIL import ImageDraw, ImageFont

from config import token

# Пути и общие параметры
dataset_path = 'data/raw/RTSD' if colab else os.path.join('data', 'raw', 'RTSD')
data_prepared_path = '../content/drive/MyDrive/TSR/data/prepared' if colab else os.path.join('data', 'prepared')
models_path = '../content/drive/MyDrive/TSR/models' if colab else 'models'

device_id = 0
if torch.cuda.is_available() == True:
    device = f'cuda:{device_id}'
#elif torch.backends.mps.is_available() == True:
#    device = 'mps'
else:
    device = 'cpu'

detector_name = 'resnet50_v2_augmented_b8'
detector_epoch = 5       # эпоха детектора с лучшей метрикой (recall)

classifier_name = 'resnet152_add_signs_bg100_tvs_randomchoice_perspective_colorjitter_resizedcrop_erasing_adam_001_sh_10_06'
classifier_epoch = 29  # эпоха классификатора с лучшей метрикой (f1)

threshold = 0.9
debug_mode = False

# загрузка моделей
def load_model_detection(detector_name, num_classes, epoch):
    # load model
    detector = None

    if ('resnet50_v2' in detector_name) or ('resnet50v2' in detector_name):
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    elif 'resnet50' in detector_name:
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    elif 'mobilenet_v3_320' in detector_name:
        detector =torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    elif 'mobilenet_v3' in detector_name:
        detector =torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    if detector is None:
        print("Неверно указано название детектора")
    else:
        print(f'Загружен детектор {detector_name}')
        
        # get number of input features for the classifier
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Загрузка весов модели
        checkpoint = torch.load(os.path.join(models_path, f'chkpt_detector_{detector_name}_{epoch}.pth'), map_location=device)
        detector.load_state_dict(checkpoint['model_state_dict'])
        print(f"Для детектора {detector_name} загружены веса эпохи {epoch}")
    
    return detector

def load_model_classifier(classifier_name, num_classes, epoch):
    if 'resnet152' in classifier_name:
        classifier = torchvision.models.resnet152(weights=None)
        for param in classifier.parameters():
            param.requires_grad = False
        
        classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
        for param in classifier.fc.parameters():
            param.requires_grad = True
        print(f"Загружен классификатор {classifier_name}")

        # Загрузка весов модели
        checkpoint = torch.load(os.path.join(models_path, f'classifier_{classifier_name}_chkpt_{epoch}.pth'), map_location=device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        print(f"Для классификатора {classifier_name} загружены веса эпохи {epoch}")
    return classifier

detector = load_model_detection(detector_name, num_classes=2, epoch=detector_epoch).to(device)
classifier = load_model_classifier(classifier_name, num_classes=156, epoch=classifier_epoch).to(device)

# Запуск бота
image = None
img_test = None
result = None

bot = telebot.TeleBot(token)

def get_prediction(img, threshold):

    # Загрузка и преобразоваие изображения в тензор
    transforms_img=torchvision.transforms.ToTensor()
    img_transformed = transforms_img(img).to(device)
    
    # Детекция знаков
    global detector
    detector.eval()
    prediction = detector([img_transformed])
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(prediction[0]['boxes'].detach().cpu().numpy())]
    #pred_labels = list(prediction[0].get('labels').cpu().numpy())
    pred_scores = list(prediction[0].get('scores').detach().cpu().numpy())
    
    try:
        pred_tr = [pred_scores.index(x) for x in pred_scores if x > threshold][-1]
        #pred_labels = pred_labels[:pred_tr+1]
        pred_boxes = pred_boxes[:pred_tr+1]
        pred_scores = pred_scores[:pred_tr+1]    
    except:
        pred_boxes = []
        pred_scores = []

    # Классификация знаков
    global classifier
    classifier.eval()
    transforms_sign = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                                      torchvision.transforms.ToTensor()
                                                      ]) 
    with open(os.path.join(dataset_path, 'label_map.json'), 'r') as read_file:
        label_map = json.load(read_file)
    read_file.close()
    label_map = {v:k for k, v in label_map.items()}
     
    pred_labels = []
    for i in range(len(pred_boxes)):
        sign = img.crop(pred_boxes[i])
        sign = transforms_sign(sign).to(device)
        pred_label = int(classifier(sign.unsqueeze(0)).data.max(1,keepdim=True)[1][0][0])
        pred_label = label_map.get(pred_label)
        pred_labels.append(pred_label)
    return pred_boxes, pred_labels, pred_scores

@bot.message_handler(content_types=['text', 'photo'])
def get_text_message(message):
    global image
    global img_test
    global result
    global threshold
    global debug_mode
    if message.text:
        if re.search(r'0[\.,\s]\d+', message.text):
            digit_board = re.search('0[\.,\s]\d+', message.text).span()
            threshold = message.text[digit_board[0] : digit_board[1]]
            threshold = re.sub(r',|\s', '.', threshold)
            threshold = round(float(threshold), 2)
            bot.send_message(message.from_user.id, f'Установлен threshold = {threshold}')
        
        elif re.search('трешхолд|трэшхолд|threshold', message.text.lower()):
            bot.send_message(message.from_user.id, f'Threshold = {threshold}')

        elif re.search('debug|дебаг', message.text.lower()):
            debug_mode = not debug_mode
            bot.send_message(message.from_user.id, f'Установлен debug_mode = {debug_mode}')
        
        else:
            bot.send_message(message.from_user.id, 'Привет! Бот принимает фотографию с дорожными знаками и возвращает фотографию с отмеченными дорожными знаками и их названиями.')
    
    elif message.photo:
        image = message.photo
        file_path = '.'
        raw = message.photo[3].file_id
        name = raw + '.jpg'
        file_info = bot.get_file(raw)
        downloaded_file = bot.download_file(file_info.file_path)
        with open (name, 'wb') as new_file:
            new_file.write(downloaded_file)
        img = Image.open(name, 'r')
        img_test = img
        
        result = get_prediction(img, threshold=threshold)
                
        new_image = img_test.copy()
        font = ImageFont.load_default()
        pencil = ImageDraw.Draw(new_image)
        for i in range(len((result[0]))):
            pencil.rectangle(result[0][i], fill = None, width=2, outline='yellow')
            text_x = result[0][i][0]
            text_y = result[0][i][1]
            mark = str(result[1][i]) + ': ' + str(round(result[2][i], 2))
            pencil.text((text_x, text_y - 9), mark, font=font, fill = 'red', size = 20)
        
        with open(os.path.join(data_prepared_path, 'labels_names_map.json'), 'r') as read_file:
            labels_names_map = json.load(read_file)
        read_file.close()

        labels = []
        if result[1] != []:
            for res in result[1]:
                labels.append(res)
        labels = list(set(labels))
        labels.sort()
        labels_names = []
        for label in labels:
            labels_names.append(f"{label}: {labels_names_map.get(label)}")
        
        labels_names = '\n'.join(labels_names)

        #bot.send_message(message.from_user.id, str(result))
        bot.send_photo(message.from_user.id, new_image)
        bot.send_message(message.from_user.id, labels_names)
        
bot.polling(none_stop=True, interval=0)