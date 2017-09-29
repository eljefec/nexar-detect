from __future__ import print_function
from keras_frcnn_lib.test_frcnn import FRCNNTester
from challenge2_evaluation.evaluate.eval_challenge import eval_detector_csv
from nexet_to_pascal_voc import DatasetBuilder
import os
import cv2
import site

HEADER = 'image_filename,x0,y0,x1,y1,label,confidence'
NUM_ROIS = 32

def detect_folder(model, img_folder, dt_csv):
    count = 0
    with open(dt_csv, 'w') as f:
        print(HEADER, file=f)
        for img_name in os.listdir(img_folder):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print('{}: {}'.format(count, img_name))
            filepath = os.path.join(img_folder, img_name)

            img = cv2.imread(filepath)
            pred_boxes = model.predict(img)

            for box in pred_boxes:
                print('{},{},{},{},{},{},{}'.format(img_name, box.x1, box.y1, box.x2, box.y2, box.class_name, box.prob), file=f)

            count += 1

def detect_val():
    site.addsitedir('./keras_frcnn_lib')
    for bbox_threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        frcnn = FRCNNTester('config.pickle', NUM_ROIS)
        dt_csv = '/home/eljefec/data/nexet/dt/val_dt_frcnn_bb{}.csv'.format(bbox_threshold)
        detect_folder(frcnn, '/home/eljefec/data/nexet/val', dt_csv)
        yield dt_csv

def try_detect_val():
    with open('exp_frcnn_bbox.txt', 'w') as write_f:
        gt_csv = '/home/eljefec/data/nexet/val_gt.csv'
        for dt_csv in detect_val():
            iou_threshold = 0.75
            ap = eval_detector_csv(gt_csv, dt_csv, iou_threshold)
            report = '{}: {}'.format(dt_csv, ap)
            print(report, file=write_f)
            print(report)

def generate_groundtruth(pascal_csv, gt_csv):
    dataset = DatasetBuilder()
    dataset.read_from_pascal_voc(pascal_csv)

    with open(gt_csv, 'w') as f:
        print(HEADER, file=f)
        for filename, ex in dataset.examples.iteritems():
            print(filename)
            for box in ex.boxes:
                print('{},{},{},{},{},{},{}'.format(ex.filename, box.x1, box.y1, box.x2, box.y2, box.class_name, 1.0), file=f)

def groundtruth_val():
    generate_groundtruth('/home/eljefec/data/nexet/val_pascal.csv', '/home/eljefec/data/nexet/val_gt.csv')

if __name__ == '__main__':
    # detect_val()
    # groundtruth_val()
    try_detect_val()
