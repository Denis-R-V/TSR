# Traffic-sign recognition
Система распознавания дорожных знаков на датасете RTSD (российские дорожные знаки)

Файлы:
- "Презентация проекта.pdf" - описание проекта
- 1.dataset_analysis.ipynb - анализ датасета RTSD (российские дорожные знаки)
- 2.simple_rcnn.ipynb - базовый вариант обучения детектора Faster R-CNN (детекция 155 классов)
- 3.1.detector_train.ipynb - обучение детектора 1 класса (знак) для подачи его в классификатор
- 3.2.detector_eval.ipynb - инференс детектора и анализ его метрик
- 4.1.classifier_make_background.ipynb - отбор FP детекций, которые будут искользоваться как нулевой класс (фон)
- 4.2.make_additional_datasets.ipynb - обогащение обучающего датасета знаками из других датасетов
- 4.3.classifier_train.ipynb - обучение классификатора дорожных знаков
- 4.4.classifier_eval.ipynb - инференс классификатора и анализ его метрик
- 5.evaluation.ipynb - пайплайн детоктор-классификатор, подбор threshold, расчет метрик
- 6.1.inference_local.ipynb - локальный инференс с визуализацией с помощью библиотеки FiftyOne
- 6.2.inference_bot.ipynb - запуск tg-бота

Необходимые данные:
- датасет Russian traffic sign images dataset https://www.kaggle.com/datasets/watchman/rtsd-dataset/data - распаковать в data/RTSD; изображения переместить из каталога rtsd-dataset/rtsd-dataset в каталог rtsd-dataset

Дополнительные данные:
- датасет German Traffic Sign Recognition Benchmark (GTSRB) https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign - распаковать в data/GTSRB
- датасет BelgiumTS Dataset https://www.kaggle.com/datasets/mahadevkonar/belgiumts-dataset - распаковать в data/BelgiumTS
- датасет Chinese Traffic Signs https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs - распаковать в data/ChineseTS