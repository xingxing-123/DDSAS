import os
import random
import shutil

dataset_path = 'data/imagenet'
split_dataset_path = 'data/imagenet_split'
images_num = 1281167
train_portion = 0.1
val_portion = 0.025
CLASSES = 1000

random.seed(0)


def get_image_list(path):
    imgs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.JPEG' in file:
                imgs.append(file)
    return imgs


def split():
    assert os.path.exists(dataset_path)

    train_path = os.path.join(dataset_path, 'train')
    # val_path = os.path.join(dataset_path, 'val')
    split_train_path = os.path.join(split_dataset_path, 'train')
    split_val_path = os.path.join(split_dataset_path, 'val')

    if not os.path.exists(split_dataset_path):
        os.makedirs(split_train_path)
        os.makedirs(split_val_path)

    for fn in os.listdir(train_path):
        imgs = get_image_list(os.path.join(train_path, fn))

        train_num_per_class = int(len(imgs) * train_portion)
        val_num_per_class = int(len(imgs) * val_portion)
        # print(train_num_per_class, val_num_per_class)

        print('{}, imgs num: {}, train: {}, val: {}'.format(fn, len(imgs), train_num_per_class, val_num_per_class))
        random.shuffle(imgs)
        if not os.path.exists(os.path.join(split_train_path, fn)):
            os.makedirs(os.path.join(split_train_path, fn))
        for i in range(train_num_per_class):
            source = os.path.join(train_path, fn, imgs[i])
            target = os.path.join(split_train_path, fn, imgs[i])
            shutil.copy(source, target)
        if not os.path.exists(os.path.join(split_val_path, fn)):
            os.makedirs(os.path.join(split_val_path, fn))
        for i in range(train_num_per_class, train_num_per_class + val_num_per_class):
            source = os.path.join(train_path, fn, imgs[i])
            target = os.path.join(split_val_path, fn, imgs[i])
            shutil.copy(source, target)


def check():
    import warnings
    from PIL import Image
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    paths = [train_path, val_path]
    count = 0
    for path in paths:
        for fn in os.listdir(path):
            imgs = get_image_list(os.path.join(path, fn))
            for img in imgs:
                img_path = os.path.join(path, fn, img)
                try:
                    img = Image.open(img_path)
                    count += 1
                    if count % 1000 == 0:
                        print(count)
                except:
                    print('corrupt img', img_path)


if __name__ == "__main__":
    split()