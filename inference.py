# Система распознавания дорожных знаков на датасете RTSD
# Инференс (Telegram Bot)

import os
import re

import telebot
import torch

from config import token
from src.execute import Builder

# Запуск в colab или локально
# если работаем в колабе - монтируем диск
try:
    from google.colab import drive
    drive.mount('/content/drive')
    colab=True
except:
    colab=False

# Пути к данным и параметры
device_id = 0
device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

data_prepared_path = '../content/drive/MyDrive/TSR/data/prepared' if colab else os.path.join('data', 'prepared')
models_path = '../content/drive/MyDrive/TSR/models' if colab else os.path.join('models')
images_path = '../content/drive/MyDrive/TSR/images/telebot_images' if colab else os.path.join('images', 'telebot_images')
if not os.path.exists(images_path): os.makedirs(images_path)

detector_file = 'chkpt_detector_resnet50_v2_augmented_b8_5.pth'
classifier_file = 'classifier_resnet152_add_signs_bg100_tvs_randomchoice_perspective_colorjitter_resizedcrop_erasing_adam_001_sh_10_06_model_29.pth'

detector_threshold = 0.9
classifier_threshold = 0.9
debug_mode = False

# Загрузка модели (детектор и классификатор)
model = Builder(device=device,
                detector_path=os.path.join(models_path, detector_file),
                classifier_path=os.path.join(models_path, classifier_file),
                detector_threshold=detector_threshold,
                classifier_threshold=classifier_threshold,
                debug_mode=debug_mode)

# Запуск бота
bot = telebot.TeleBot(token)

@bot.message_handler(content_types=['text', 'photo'])
def get_text_message(message):
    if message.text:                                    # to do если в тексте есть классификатор и детектор - ставим оба, если один - по одному, если не одного - запрос 1 или 2 (оба 3)
        if re.search(r'0[\.,\s]\d+', message.text):
            digit_board = re.search('0[\.,\s]\d+', message.text).span()
            threshold = message.text[digit_board[0] : digit_board[1]]
            threshold = re.sub(r',|\s', '.', threshold)
            threshold = round(float(threshold), 2)
            bot.send_message(message.from_user.id, f'Установлен threshold = {model.detector_threshold}')
        
        elif re.search('трешхолд|трэшхолд|threshold', message.text.lower()):
            bot.send_message(message.from_user.id, f'Threshold = {model.detector_threshold}')

        elif re.search('debug|дебаг', message.text.lower()):
            model.debug_mode = not model.debug_mode
            bot.send_message(message.from_user.id, f'Установлен debug_mode = {model.debug_mode}')
        
        else:
            bot.send_message(message.from_user.id, 'Привет!\nБот принимает фотографию и возвращает фотографию с отмеченными дорожными знаками и их названиями.')
    
    elif message.photo:
        raw = message.photo[3].file_id
        name = raw + '.jpg'
        file_info = bot.get_file(raw)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(os.path.join(images_path, name), 'wb') as new_file:
            new_file.write(downloaded_file)
        
        img_pred, description = model.predict_single_visualized(os.path.join(images_path, name))
        labels_names = '\n'.join(description)

        bot.send_photo(message.from_user.id, img_pred)
        bot.send_message(message.from_user.id, labels_names)
        
bot.polling(none_stop=True, interval=0)