from __future__ import print_function
from keras_frcnn_lib.test_frcnn import FRCNNTester
from challenge2_evaluation.evaluate.eval_challenge import eval_detector_csv
from nexet_to_pascal_voc import DatasetBuilder
import os
import cv2
import site
import time

HEADER = 'image_filename,x0,y0,x1,y1,label,confidence'
NUM_ROIS = 32

def flush(f):
    f.flush()
    os.fsync(f.fileno())

def detect_folder(model, img_folder, dt_csv, bbox_threshold):
    count = 0
    prog_file = 'detect_progress.txt'
    with open(prog_file, 'w') as prog_f:
        with open(dt_csv, 'w') as dt_f:
            print(HEADER, file=dt_f)
            for img_name in os.listdir(img_folder):
                if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                    continue
                filepath = os.path.join(img_folder, img_name)

                imread_start = time.time()
                img = cv2.imread(filepath)
                pred_start = time.time()
                pred_boxes = model.predict(img, bbox_threshold)
                pred_end = time.time()
                if count % 100 == 0:
                    report = '[{:.2f}, {:.2f}] {}: {}'.format(pred_start - imread_start,
                                                              pred_end - pred_start,
                                                              count, img_name)
                    print(report, file=prog_f)
                    flush(prog_f)
                    flush(dt_f)
                    print(report)

                for box in pred_boxes:
                    print('{},{},{},{},{},{},{}'.format(img_name, box.x1, box.y1, box.x2, box.y2, box.class_name, box.prob), file=dt_f)

                count += 1

def gen_detect_frcnn(val, config):
    if val:
        img_folder = '/home/eljefec/data/nexet/val'
        name = 'val'
    else:
        img_folder = '/home/eljefec/data/nexet/test'
        name = 'test'

    site.addsitedir('./keras_frcnn_lib')
    frcnn = FRCNNTester(config, NUM_ROIS)
    for bbox_threshold in [0.5]:
        dt_csv = '/home/eljefec/data/nexet/dt/{}_dt_frcnn_bb{}_{}.csv'.format(name, bbox_threshold, os.path.basename(config))
        detect_folder(frcnn, img_folder, dt_csv, bbox_threshold)
        yield dt_csv

def gen_detect_rfcn():
    site.addsitedir('./RFCN_tensorflow')
    from RFCN_tensorflow.test import RFCNTester
    for bbox_threshold in [0.0, 0.1]:
        rfcn = RFCNTester('RFCN_tensorflow/save/save', None, bbox_threshold)
        dt_csv = '/home/eljefec/data/nexet/dt/val_dt_rfcn_bb{}.csv'.format(bbox_threshold)
        detect_folder(rfcn, '/home/eljefec/data/nexet/val', dt_csv, bbox_threshold)
        yield dt_csv

def try_detect_val(dt_csv_gen, report_filename):
    iou_threshold = 0.75
    with open(report_filename, 'w') as write_f:
        gt_csv = '/home/eljefec/data/nexet/val_gt.csv'
        for dt_csv in dt_csv_gen:
            ap = eval_detector_csv(gt_csv, dt_csv, iou_threshold)
            report = '{}: {}'.format(dt_csv, ap)
            print(report, file=write_f)
            write_f.flush()
            os.fsync(write_f.fileno())
            print(report)

def detect_frcnn_test():
    gen = gen_detect_frcnn(val = False, config = 'config/config.e430.pickle')
    for item in gen:
        print('Generated {}'.format(item))

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
    # try_detect_val(gen_detect_frcnn(val = True), 'exp_frcnn_bbox0.0.txt')
    # try_detect_val(gen_detect_rfcn(), 'exp_rfcn_bbox.txt')
    detect_frcnn_test()
