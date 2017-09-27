from __future__ import print_function
from labelImg.libs.pascal_voc_io import PascalVocWriter, PascalVocReader
import cv2
import numpy as np
import os

def invalid_bbox(x1, y1, x2, y2):
    area = abs(x1 - x2) * abs(y1 - y2)
    return area < 70

class Box:
    def __init__(self, class_name, x1, y1, x2, y2):
        self.class_name = class_name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        assert(x1 <= x2)
        assert(y1 <= y2)
        self.w = x2 - x1
        self.h = y2 - y1

    def is_invalid(self):
        area = abs(self.x1 - self.x2) * abs(self.y1 - self.y2)
        return area < 70

class Example:
    def __init__(self, filename, fullpath, img_shape, boxes):
        self.filename = filename
        self.fullpath = fullpath
        self.img_shape = img_shape
        (rows, cols, depth) = img_shape[:3]
        self.width = cols
        self.height = rows
        self.depth = depth
        self.boxes = boxes

def write_example_file(examples, filepath):
    with open(filepath, 'w') as f:
        for ex in examples:
            basename = os.path.splitext(ex.filename)[0]
            print(basename, file = f)

class DatasetBuilder:
    def __init__(self):
        self.examples = {}
        self.foldername = None

    def _add_box(self, filename, fullpath, box):
        if filename in self.examples:
            ex = self.examples[filename]
        else:
            img = cv2.imread(fullpath)
            ex = Example(filename,
                         fullpath,
                         img.shape,
                         boxes = [])
            self.examples[filename] = ex
        ex.boxes.append(box)

    def split_train_test(self, val_count, val_file, train_file):
        assert(val_count < len(self.examples))

        examples = []
        for filename, ex in self.examples.iteritems():
            examples.append(ex)
        examples = np.random.permutation(examples)
        val_ex = examples[:val_count]
        train_ex = examples[val_count:]

        write_example_file(val_ex, val_file)
        write_example_file(train_ex, train_file)

    def read_from_nexet(self, img_boxes_csv, img_folder):
        self._read_from_generator(self._gen_from_nexet(img_boxes_csv, img_folder))

    def read_from_pascal_voc(self, pascal_csv):
        self._read_from_generator(self._gen_from_pascal_voc(pascal_csv))

    def write_to_pascal_voc(self, pascal_csv, pascal_folder):
        assert(len(self.examples) > 0)
        assert(self.foldername is not None)

        if not os.path.exists(pascal_folder):
            print('Making {}'.format(pascal_folder))
            os.makedirs(pascal_folder)

        print('Writing {} examples to pascal voc.'.format(len(self.examples)))

        with open(pascal_csv, 'w') as write_f:
            print(self.foldername, file = write_f)

            for filename, ex in self.examples.iteritems():
                writer = PascalVocWriter(self.foldername, ex.filename, ex.img_shape, databaseSrc = 'Nexet', localImgPath = ex.fullpath)
                for box in ex.boxes:
                    writer.addBndBox(xmin = box.x1,
                                     ymin = box.y1,
                                     xmax = box.x2,
                                     ymax = box.y2,
                                     name = box.class_name,
                                     difficult = False)
                pascal_path = os.path.join(pascal_folder, os.path.splitext(ex.filename)[0] + '.xml')
                writer.save(pascal_path)

                print(pascal_path, file = write_f)

    def _gen_from_pascal_voc(self, pascal_csv):
        with open(pascal_csv, 'r') as f:
            self.foldername = f.readline().strip()

            for line in f:
                line = line.strip()
                reader = PascalVocReader(line)
                fullpath = os.path.join(self.foldername, os.path.splitext(os.path.basename(reader.filepath))[0] + '.jpg')
                for shape in reader.getShapes():
                    class_name = shape[0]
                    points = shape[1]
                    x1 = points[0][0]
                    y1 = points[0][1]
                    x2 = points[2][0]
                    y2 = points[2][1]
                    box = Box(class_name, x1, y1, x2, y2)
                    yield reader.filepath, fullpath, box

    def _gen_from_nexet(self, img_boxes_csv, img_folder):
        assert(os.path.exists(img_folder))

        self.foldername = img_folder

        with open(img_boxes_csv, 'r') as read_f:
            line_count = 0
            for line in read_f:
                line_count += 1
                line_split = line.strip().split(',')
                (filename, x1, y1, x2, y2, class_name, confidence) = line_split
                fullpath = os.path.join(img_folder, filename)
                if ',' in fullpath:
                    print('Warning: Path contains comma:', fullpath, 'line:', line)
                else:
                    try:
                        x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                        box = Box(class_name, x1, y1, x2, y2)
                        yield filename, fullpath, box
                    except ValueError:
                        print('Warning: Failed to make box. line: [{}]'.format(line.strip()))

    def _read_from_generator(self, gen):
        for item in gen:
            filename = item[0]
            fullpath = item[1]
            box = item[2]
            if box.is_invalid():
                print('Warning: Rejecting invalid bbox')
            else:
                if os.path.exists(fullpath):
                    self._add_box(filename, fullpath, box)
                else:
                    print('Warning: Non-existent file:', fullpath)

def nexet_to_pascal_voc():
    builder = DatasetBuilder()
    print('read_from_nexet')
    builder.read_from_nexet('/home/eljefec/data/nexet/train_boxes.csv',
                            '/home/eljefec/data/nexet/train')

    builder.split_train_test(val_count=1000,
                             val_file='/home/eljefec/data/nexet/ImageSets/Main/test.txt',
                             train_file='/home/eljefec/data/nexet/ImageSets/Main/trainval.txt')

#    print('write_to_pascal_voc')
#    builder.write_to_pascal_voc('/home/eljefec/data/nexet/train_pascal.csv',
#                                '/home/eljefec/data/nexet/train_pascal')
#
#    print('{} read from nexet'.format(len(builder.examples)))

#    other_builder = DatasetBuilder()
#    print('read_from_pascal_voc')
#    other_builder.read_from_pascal_voc('/home/eljefec/data/nexet/train_pascal.csv')
#    print('{} read from pascal'.format(len(other_builder.examples)))

if __name__ == '__main__':
    nexet_to_pascal_voc()
