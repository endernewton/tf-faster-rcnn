'''
# Generate image with random string & font & size
1. select font
1. select size
1. select chars 2 ~ 10
1. pick random position
1. check fill_image_ratio

'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from xml.etree.ElementTree import Element, SubElement, ElementTree, dump

import random
import copy
from string import ascii_lowercase, ascii_uppercase
from os import listdir, path, makedirs
from os.path import isfile, isdir, join, basename, exists

import os
import math
begin = 0xac00
end = 0xd7a3

POINT_PER_CELL = 10

NUM_OF_NONE = 0

class Sample_IMG():
    def __init__(self, fill_ratio=0.3, width=800, height=600):
        self.fill_ratio = fill_ratio
        self.width = width
        self.height = height
        self.fill_area = 0

        self.chars = list()
        self.char_positions = list()
        self.boxes = list()

        '''
        # store boxes by descending order
        1. begin with ((0, 0), (N, M))
        2. if place box in ((i, j), (n, m)) where (i, j > 0 and n < N, m < M and i < n, j < m)
        3. split remaining areas into 4 boxes: ((0, 0), (N, j)), ((0, j), (i, m)), ((n, j), (N, m)), ((0, m), (N, M))
        '''
        self.N = int(self.width / POINT_PER_CELL)
        self.M = int(self.height / POINT_PER_CELL)
        self.areas = [(0, 0, self.N, self.M)]

    def hasFill(self):
        if self.fill_ratio * self.width * self.height < self.fill_area:
            return True
        else:
            return False

    def appendChars(self, chDataset, font, size, chars):
        imFont = chDataset.getIMFont(font, size)
        char_sizes = [imFont.getsize(ch) for ch in chars]
        sum_area = sum([ch_size[0] * ch_size[1] for ch_size in char_sizes])

        self.chars.append((font, size, chars, char_sizes))
        self.fill_area += sum_area

    def assignBox(self, char_width, char_height):
        '''

        1. get one box from self.area
        2. check is it fit to box

        :param char_width:
        :param char_height:
        :return:
        '''
        char_width_n = math.ceil(char_width / POINT_PER_CELL)
        char_height_m = math.ceil(char_height / POINT_PER_CELL)

        assigned_box = None

        new_boxes = []
        for idx, box in enumerate(self.areas):
            zero_x = box[0]
            zero_y = box[1]
            N = box[2]
            M = box[3]
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            # char_box not fit into box, skip it
            if box_width < char_width_n or box_height < char_height_m:
                continue

            if box_width == char_width_n:
                i = zero_x
            else:
                i = zero_x + random.randrange(box_width - char_width_n)

            if box_height == char_height_m:
                j = zero_y
            else:
                j = zero_y + random.randrange(box_height - char_height_m)

            # x_offset = random.randrange(POINT_PER_CELL)
            # y_offset = random.randrange(POINT_PER_CELL)
            x_offset = 0
            y_offset = 0

            n = i + char_width_n
            m = j + char_height_m

            assigned_box = (i * POINT_PER_CELL + x_offset, j * POINT_PER_CELL + y_offset, n * POINT_PER_CELL , m * POINT_PER_CELL)

            new_boxes = [box for box in [(zero_x, zero_y, N, j),
                         (zero_x, j, i, m),
                         (n, j, N, m),
                         (zero_x, m, N, M)] if box[0] < box[2] and box[1] < box[3]]
            break

        self.areas = self.areas[0:idx] + self.areas[idx+1:] + new_boxes

        return assigned_box

    def placeStoreChars(self):
        # sort by area of chars, descending order
        self.chars.sort(key=lambda x: x[1], reverse=True)

        for char_info in self.chars:
            font = char_info[0]
            font_size = char_info[1]
            chars = char_info[2]
            char_sizes = char_info[3]
            char_width = sum([ch_size[0] for ch_size in char_sizes])
            char_height = max([ch_size[1] for ch_size in char_sizes])

            box = self.assignBox(char_width, char_height)
            if box is None:
                x = random.randrange(self.width - char_width)
                y = random.randrange(self.height - char_height)
                box = (x, y, x + char_width, y + char_height)

            self.boxes.append(box)


    def makeXML(self, char_list, img_filename, idx=None, output_path=None):
        # char_info: (x, y, font_size, font_size)

        def indent(elem, level=0):
            i = "\n" + level * "    "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "    "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    indent(elem, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        annotation = Element("annotation")
        SubElement(annotation, "folder").text = output_path.split('/')[-1]
        SubElement(annotation, "filename").text = str(img_filename.split('/')[-1])
        size = Element("size")
        annotation.append(size)
        SubElement(size, "width").text = str(self.width)
        SubElement(size, "height").text = str(self.height)
        SubElement(size, "depth").text = '3'

        for ch_info in char_list:
            obj = Element("object")
            annotation.append(obj)
            SubElement(obj, "name").text = ch_info[1]
            bndbox = Element("bndbox")
            obj.append(bndbox)

            x, y, width, height = ch_info[0]

            SubElement(bndbox, "xmin").text = str(x)
            SubElement(bndbox, "ymin").text = str(y)
            SubElement(bndbox, "xmax").text = str(width)
            SubElement(bndbox, "ymax").text = str(height)

        indent(annotation)
        ElementTree(annotation).write(os.path.join(output_path, 'annotations', '%d.xml' % idx), encoding='utf-8')

    def prn(self, chDataset=None, idx=None, output_path=None):
        '''

        :param chDataset:
        :param filename:
        :return:
        '''
        # print('fill_area', self.fill_area, 'width', self.width, 'height', self.height)
        # print('chars', [ch for ch in self.chars])
        # print('# of chars', sum([len(ch[2]) for ch in self.chars]))

        char_list = []

        # base_font = ImageFont.truetype('/Library/Fonts/AppleGothic.ttf', 10)
        if chDataset is not None:
            im = Image.new('RGB', (800, 600), color=(256, 256, 256))
            draw = ImageDraw.Draw(im)
            for box, char_info in zip(self.boxes, self.chars):
                font = char_info[0]
                font_size = char_info[1]
                chars = char_info[2]

                self.boxes.append(box)
                imFont = chDataset.getIMFont(font, font_size)
                draw.text(box[:2], ''.join(chars), font=imFont, fill=(0, 0, 0))

                # label_box = [box[0], box[1] - 10]
                # draw.text(label_box, ''.join(chars) + font.split('/')[-1], font=base_font, fill=(256, 0, 0))
                # draw.rectangle(box, outline=(256, 0, 0))

                width_offset = 0
                for ch in chars:
                    ch_width, ch_height = imFont.getsize(ch)
                    ch_box = (box[0] + width_offset, box[1], box[0] + width_offset + ch_width, box[1] + ch_height)
                    # draw.rectangle(ch_box, outline=(0, 256, 0))
                    width_offset += ch_width
                    char_list.append((ch_box, ch))

            if idx is None or output_path is None:
                import uuid
                filename = '%s.png' % uuid.uuid4()
            else:
                filename = join(output_path, 'images', '%d.png' % idx)
            im.save(filename)

            self.makeXML(char_list, filename, idx=idx, output_path=output_path)

        return sum([len(ch[2]) for ch in self.chars])

    @classmethod
    def generate(cls, chDataset, n_char=10, fill_ratio=0.2, width=800, height=600):
        '''
        generate single image sample containing multiple chars

        :param n_char: number of char per font & size
        :param ratio: fill area ratio of image with character region
        :param width: width of image
        :param height: height of image
        :return:
        '''
        if chDataset.isValid() is False:
            return None

        sampleImg = Sample_IMG(fill_ratio=fill_ratio, width=width, height=height)

        while sampleImg.hasFill() is False and chDataset.hasMoreData() is True:
            tuple_idx, chars = chDataset.getChars(n_char=n_char)
            if len(chars) > 0:
                (font, size) = chDataset.font_size_tuples[tuple_idx]
                sampleImg.appendChars(chDataset, font, size, chars)

        sampleImg.placeStoreChars()
        return sampleImg

class CH_Dataset():
    def __init__(self, font_path=None):
        self.char_list = [chr(begin + idx) for idx in range(end - begin + 1)] \
                         + [x for x in (ascii_lowercase + ascii_uppercase)] + [str(x) for x in range(10)] + [x for x in
                                                                                                             '~!@#$%^&*()_+-=<>?,.;:[]{}|']

        self.char_list = self.char_list[:100]
        self.font_list = [join(font_path, f) for f in listdir(font_path) if
                          isfile(join(font_path, f)) and f.find('.DS_Store') == -1]

        self.font_sizes = [10] * 5 + [20] * 10 + [30] * 7 + [50] * 5 + [100] * 2

        print('total # of chars', len(self.char_list) * len(self.font_list) * len(self.font_sizes))

        self.counter = 0
        self.gen_counter = 0
        self.fill_ratio = 0

        self.font_size_tuples = None
        self.char_list_list = None
        self.counter_list = None
        self.filled_counter = None
        self.filled_counter_num = 0

        self.imFont = dict()

    def isStop(self):
        self.counter = self.counter + 1
        if self.counter > 50:
            return True
        else:
            return False

    def getFont(self):
        if self.font_list is not None and len(self.font_list) > 0 and len(self.font_sizes) > 0:
            self.gen_counter += 1
            return self.font_list[self.gen_counter % len(self.font_list)], self.font_sizes[
                self.gen_counter % len(self.font_sizes)]
        else:
            return None, None

    def isValid(self):
        if self.font_size_tuples is None or self.char_list_list is None or self.counter_list is None:
            return False
        else:
            return True

    def hasMoreData(self):
        if self.filled_counter_num >= len(self.counter_list):
            return False
        else:
            return True

    def getIMFont(self, font, size):
        if font not in self.imFont:
            self.imFont[font] = dict()

        if size not in self.imFont[font]:
            self.imFont[font][size] = ImageFont.truetype(font=font, size=size)
        return self.imFont[font][size]

    def getChars(self, n_char=10):
        # pick a random char list
        idx = random.randrange(len(self.char_list_list))

        # split first k chars(1~10)
        k_chars = random.randrange(n_char) + 1

        # get fist k chars
        current_char_idx = self.counter_list[idx]

        # make sure target_idx <= len(self.char_list_list[idx])
        target_idx = current_char_idx + k_chars if current_char_idx + k_chars < len(self.char_list_list[idx]) else len(
            self.char_list_list[idx])

        # pick chars (current_char_idx:target_idx)
        self.counter_list[idx] = target_idx

        if target_idx >= len(self.char_list_list[idx]):
            self.filled_counter[idx] = True
            self.filled_counter_num = sum(self.filled_counter)
        return idx, self.char_list_list[idx][current_char_idx:target_idx]

    def generateSamples(self, n_char=7, fill_ratio=0.2, width=800, height=600, output_path=None):
        '''
        generate samples:
            1. generate (font, size) tuple
            2. create N char list(N = # of (font, size) tuple)
            3. shuffle each char list
            4. create

        :param n_char: number of char per sentence
        :param ratio: fill area ratio of image with character region
        :param width: width of image
        :param height: height of image

        :return:
        '''
        self.font_size_tuples = [(f, s) for f in self.font_list for s in self.font_sizes]
        self.char_list_list = [copy.deepcopy(self.char_list) for _ in range(len(self.font_size_tuples))]
        [random.shuffle(ch_list) for ch_list in self.char_list_list]
        self.counter_list = [0] * len(self.char_list_list)

        self.filled_counter = [counter >= len(char_list) for counter, char_list in zip(self.counter_list, self.char_list_list)]

        total_chars = 0
        results = list()

        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if not os.path.exists(join(output_path, 'images')):
                os.makedirs(join(output_path, 'images'))
            if not os.path.exists(join(output_path, 'annotations')):
                os.makedirs(join(output_path, 'annotations'))

        output_idx = 0
        while self.hasMoreData() is True:
            gen_img = Sample_IMG.generate(self, n_char=n_char, fill_ratio=fill_ratio, width=width, height=height)
            if gen_img:
                total_chars += gen_img.prn(chDataset=self, idx=output_idx, output_path=output_path)
                results.append(gen_img)
                output_idx += 1

        NUMBER_OF_IMAGES = output_idx
        # split train/val/test set
        shuffled_index = list(range(NUMBER_OF_IMAGES))
        random.shuffle(shuffled_index)

        # splitting train/validation/test set (unit: %)
        TRAIN_SET = 80
        VALID_SET = 10
        TEST_SET = 10
        num_train = int(NUMBER_OF_IMAGES * TRAIN_SET / (TRAIN_SET + VALID_SET + TEST_SET))
        num_valid = int(NUMBER_OF_IMAGES * VALID_SET / (TRAIN_SET + VALID_SET + TEST_SET))
        num_test = NUMBER_OF_IMAGES - num_train - num_valid

        with open(join(output_path, 'train.txt'), "w") as wf:
            for index in shuffled_index[0:num_train]:
                wf.write(str(index) + '\n')

        with open(join(output_path, 'val.txt'), "w") as wf:
            for index in shuffled_index[num_train:num_train + num_valid]:
                wf.write(str(index) + '\n')

        with open(join(output_path, 'trainval.txt'), "w") as wf:
            for index in shuffled_index[0:num_train + num_valid]:
                wf.write(str(index) + '\n')

        with open(join(output_path, 'test.txt'), "w") as wf:
            for index in shuffled_index[num_train + num_valid:]:
                wf.write(str(index) + '\n')

        with open(join(output_path, 'labels.txt'), "w") as wf:
            for label in self.char_list:
                wf.write(str(label) + '\n')

        print("Train / Valid / Test : {} / {} / {}".format(num_train, num_valid, num_test))
        print("Output path: {}".format(output_path))

        return results


chd = CH_Dataset(font_path='fonts')
chd.generateSamples(output_path='data/fontdataset')
