import os
import random
import shutil

def organize_folder(folder):
    _, _, filenames = next(os.walk(folder))
    unique_classes = {filename.split(".")[0] for filename in filenames}
    for _class in unique_classes:
        path = os.path.join(folder, _class)
        if not os.path.exists(path):
            os.makedirs(path)
        for filename in filenames:
            if filename.startswith(_class):
                shutil.move(os.path.join(folder, filename), os.path.join(path, filename))        
    
def create_sample_folder(_from, to, percentage=0.1, move=True):
    if not os.path.exists(to):
        os.makedirs(to)
    _, folders, _ = next(os.walk(_from))
    for folder in folders:
        if not os.path.exists(os.path.join(to, folder)):
            os.makedirs(os.path.join(to, folder))
        _, _, files = next(os.walk(os.path.join(_from, folder)))
        sample = random.sample(files, int(len(files) * percentage))
        for filename in sample:
            if move:
                shutil.move(os.path.join(_from, folder, filename), os.path.join(to, folder, filename))
            else:
                shutil.copyfile(os.path.join(_from, folder, filename), os.path.join(to, folder, filename))


