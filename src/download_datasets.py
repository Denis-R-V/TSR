import os
import shutil
import kaggle

kaggle.api.authenticate()

def download_RTSD(path):
    kaggle.api.dataset_download_files('watchman/rtsd-dataset', path=path, unzip=True)

    for file in os.listdir(os.path.join(path, 'rtsd-frames', 'rtsd-frames')):
        # Source path 
        source = os.path.join(path, 'rtsd-frames', 'rtsd-frames', file)
        
        # Destination path 
        destination = os.path.join(path, 'rtsd-frames', file)
        
        # Move the content of source to destination 
        shutil.move(source, destination) 

    os.removedirs(os.path.join(path, 'rtsd-frames', 'rtsd-frames'))

def download_GTSRB(path):
    kaggle.api.dataset_download_files('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign', path=path, unzip=True)

def download_BelgiumTS(path):
    kaggle.api.dataset_download_files('mahadevkonar/belgiumts-dataset', path=path, unzip=True)

def download_ChineseTS(path):
    kaggle.api.dataset_download_files('dmitryyemelyanov/chinese-traffic-signs', path=path, unzip=True)