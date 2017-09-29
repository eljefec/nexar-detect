from __future__ import print_function
from keras_frcnn.test_frcnn import FRCNNTester
from nexet_to_pascal_voc import DatasetBuilder
import os
import cv2

HEADER = 'image_filename,x0,y0,x1,y1,label,confidence'
NUM_ROIS = 32

def detect_folder(model, img_folder, dt_csv):
    with open(dt_csv, 'w') as f:
        print(HEADER, file=f)
        for img_name in os.listdir(img_path):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            filepath = os.path.join(img_path, img_name)

            img = cv2.imread(filepath)
            pred_boxes = model.predict(img)

            for box in pred_boxes:
                print('{},{},{},{},{},{},{}'.format(img_name, box.x1, box.y1, box.x2, box.y2, box.class_name, box.prob), file=f)

def detect_val():
    frcnn = FRCNNTester('config.pickle', NUM_ROIS)
    detect_folder(frcnn, '/home/eljefec/data/nexet/val', '/home/eljefec/data/nexet/dt/frcnn_dt.csv')

def generate_groundtruth(pascal_csv, gt_csv):
    dataset = DatasetBuilder()
    dataset.read_from_pascal_voc(self, pascal_csv)

    with open(gt_csv, 'w') as f:
        print(HEADER, file=f)
        for ex in dataset.examples:
            for box in ex.boxes:
                print('{},{},{},{},{},{},{}'.format(ex.filename, box.x1, box.y1, box.x2, box.y2, box.class_name, 1.0), file=f)

def groundtruth_val():
    generate_groundtruth('/home/eljefec/data/nexet/val_pascal.csv', '/home/eljefec/data/nexet/val_gt.csv')

if __name__ == '__main__':
    detect_val()
    groundtruth_val()
