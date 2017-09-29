from shutil import copyfile
from nexet_to_pascal_voc import write_pascal_voc_csv
import os

def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def copy_subset(image_list_file,
                img_folder,
                pascal_folder,
                dest_img_folder,
                dest_pascal_folder):

    makedirs(dest_img_folder)
    makedirs(dest_pascal_folder)

    with open(image_list_file, 'r') as f:
        for line in f:
            basename_no_ext = line.strip()

            img_filename = os.path.join(img_folder, basename_no_ext + '.jpg')
            dest_img_filename = os.path.join(dest_img_folder, basename_no_ext + '.jpg')

            copyfile(img_filename, dest_img_filename)

            pascal_filename = os.path.join(pascal_folder, basename_no_ext + '.xml')
            dest_pascal_filename = os.path.join(dest_pascal_folder, basename_no_ext + '.xml')

            copyfile(pascal_filename, dest_pascal_filename)


def copy():
    copy_subset('/home/eljefec/data/nexet/ImageSets/Main/test.txt',
                '/home/eljefec/data/nexet/train',
                '/home/eljefec/data/nexet/train_pascal',
                '/home/eljefec/data/nexet/val',
                '/home/eljefec/data/nexet/val_pascal')

def write_val_pascal_voc():
    write_pascal_voc_csv('/home/eljefec/data/nexet/val',
                         '/home/eljefec/data/nexet/val_pascal',
                         '/home/eljefec/data/nexet/val_pascal.csv')

if __name__ == '__main__':
    # copy()
    write_val_pascal_voc()
