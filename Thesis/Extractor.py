from random import Random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection
import pefile, zipfile, json, sys, re
import cv2 as cv
import matplotlib.cm as cm

from sultan.api import Sultan
from os import listdir
from os.path import isfile, join
import math


class Extractor:
    def chooseWith(self, sz):
        w = 0

        if sz > 1000:
            w = 1024
        elif sz > 500:
            w = 768
        elif sz > 200:
            w = 512
        elif sz > 100:
            w = 384
        elif sz > 60:
            w = 256
        elif sz > 30:
            w = 128
        elif sz > 10:
            w = 64
        else:
            w = 32

        return w

    def get_name(src):
        name = []
        for f in listdir(src):
            name.append(f[:f.index('.')] + '.json')
        return name

    def read_data(self, path, lent):
        data_ = []
        width = []
        height = []
        name = []
        for f in listdir(path):
            file = join(path, f)
            if isfile(file):
                with open(file, 'r') as js:
                    data = [json.loads(line) for line in js]
                    data_.append(data[0]['pixels'])
                    width.append(data[0]['width'])
                    height.append(data[0]['height'])
                    name.append(data[0]['name'])
                    js.close()
        data_ = np.array(data_)
        width = np.array(width)
        height = np.array(height)
        name = np.array(name)
        return (data_[:lent], width[:lent], height[:lent], name[:lent])

    def read_Binary(self, file):
        pixels = []
        h, w = 0, 0

        with open(file, 'r') as f:
            for line in f:
                sz = len(line)
                content = line[:sz - 1].split(' ')[1:]
                for byte in content:
                    if byte[0] == '?':
                        pixels.append(0)
                    else:
                        pixels.append(int('0x' + byte, 16))
        sz = len(pixels)
        w = self.chooseWith(sz / 1024)
        h = int(sz / w)
        return pixels[:h * w], h, w

    def read_PE(self, file):
        pixels = []
        h, w = 0, 0
        with open(file, 'rb') as f:
            content = f.read()
            two_first_byte = chr(content[0]) + chr(content[1])
            for x in content:
                pixels.append(x)
        sz = len(pixels)
        w = self.chooseWith(sz / 1024)
        h = int(sz / w)
        return pixels[: h * w], h, w

    def resizeBilinearGray(self, file, w_2, h_2, isBinary=True):
        if isBinary:
            pixels, h, w = self.readBinary(file)
        else:
            pixels, h, w = self.readPE(file)
        temp = []
        x_ratio = (int)(((w - 1) << 16) / w_2)
        y_ratio = (int)(((h - 1) << 16) / h_2)
        y = 0
        for i in range(h_2):
            yr = (int)(y >> 16)
            y_diff = y - (yr << 16)
            one_min_y_diff = 65536 - y_diff
            y_index = yr * w
            x = 0
            for j in range(w_2):
                xr = (int)(x >> 16)
                x_diff = x - (xr << 16)
                one_min_x_diff = 65536 - x_diff
                index = y_index + xr

                A = pixels[index] & 0xff
                B = pixels[index + 1] & 0xff
                C = pixels[index + w] & 0xff
                D = pixels[index + w + 1] & 0xff

                # Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + D(w)(h)
                gray = (int)((
                                 A * one_min_x_diff * one_min_y_diff + B * x_diff * one_min_y_diff + C * y_diff * one_min_x_diff + D * x_diff * y_diff) >> 32)

                temp.append(gray)
                x += x_ratio
            y += y_ratio;
        return temp

    def randomProjection(self, pixels, h, w, d):
        transfer = random_projection.SparseRandomProjection(d)
        X = np.reshape(pixels, (h, w))
        Y = transfer.fit_transform(X)
        return Y, h, d

    def display(self, pixels, w, h, n, path):
        g = np.reshape(pixels, (h, w))
        # plt.imshow(g, cmap=cm.gray)
        plt.imshow(g)
        plt.savefig(path + n + '.png')
        # plt.show()

    def entropy(self, bytes):
        p = [0.0] * 256
        for byte in bytes:
            p[byte] += 1.0
        for i in range(len(p)):
            p[i] = p[i] / len(bytes)
        entropy = 0
        for i in range(len(p)):
            if p[i] != 0.0:
                entropy += p[i] * math.log2(p[i])
        return -entropy

    def histogram_2d(self, pixels, window):
        histogram = [[0 for x in range(16)] for y in range(16)]
        start = 0
        while (start < len(pixels)):
            end = min(start + window, len(pixels))
            bytes = pixels[start: end]
            ent = self.entropy(bytes)
            ent = math.trunc(ent * 2)
            for byte in bytes:
                x = math.trunc(byte / 16)
                # print(x)
                # print(ent)
                histogram[x][ent] += 1
            start += window
        ans = []
        for i in range(16):
            for j in range(16):
                ans.append(histogram[i][j])
        return ans

    def extract_2d_histogram(self, src, dst, window):
        cnt = 0
        for f in listdir(src):
            file = join(src, f)
            try:
                with open(file, 'r') as js:
                    data = [json.loads(line) for line in js]
                    pixels = data[0]['pixels']
                    name = data[0]['name']
                    ent = self.entropy(pixels)
                    histogram = self.histogram_2d(pixels, window)
                    # s = sum(histogram)
                    # histogram = [x / s for x in histogram]
                    with open(dst + f, 'w') as j:
                        data = {'name': name, 'feature': histogram}
                        json.dump(data, j)
                        j.close()
                        cnt += 1
                        print("file's id: %d, name = %s" % (cnt, name))
                    js.close()
            except Exception:
                print('ERROR: ' + file)

    def build_one_gram_feature(self, src, dst):
        cnt = 0
        bytes = [0] * 255
        for f in listdir(src):
            file = join(src, f)
            with open(file, 'r') as js:
                data = [json.loads(line) for line in js]
                name = data[0]['name']
                pixels = data[0]['pixels']
                for x in pixels:
                    bytes[x] += 1
                with open(dst + name + '.json') as j:
                    cnt += 1
                    data = {'name': name, 'feature': bytes}
                    json.dump(data, j)
                    print("file's id: %d, name = %s" % (cnt, name))

    def extract_dll(self, src, dst):
        cnt = 0
        for f in listdir(src):
                if isfile(dst + f + '.json'):
                    continue
                file = join(src, f)
                dll = self.extract_import_lib(src + f)
                if len(dll) == 0:
                    continue
                cnt += 1
                with open(dst + f + '.json', 'w') as js:
                    data = {'name': f, 'feature': dll}
                    json.dump(data, js)
                print("file's id : %d, name= %s" % (cnt, f))

    def extract_import_lib(self, file):
        imps = []
        try:
            pe = pefile.PE(file)
            pe.parse_data_directories()
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name == None:
                        continue
                    imps.append(str(imp.name, "latin"))
        except Exception:
            print("not PE file")
        return imps

    def extract_image_file_PE(self, src, dst):
        cnt = 0
        for f in listdir(src):
            if isfile(dst+f):
                continue
            file = join(src, f)
            try:
                pe = pefile.PE(file)
                with open(file, 'rb') as ff:
                    content = ff.read()
                    if len(content) == 0:
                        continue
                    two_first_byte = chr(content[0]) + chr(content[1])
                    if two_first_byte != "MZ":
                        continue
                    pixels = []
                    h, w = 0, 0
                    for byte in content:
                        pixels.append(byte)
                    sz = len(pixels)
                    w = self.chooseWith(sz / 1024)
                    h = int(sz / w)
                    with open(dst + f + '.json', 'w') as js:
                        data = {'name': f, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                        json.dump(data, js)
                        js.write('\n')
                        cnt += 1
                        print("file's id: %d, name = %s" % (cnt, f))
            except Exception:
                print("not PE file")

    def extract_image(self, zip_file, out_file):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            for name in zip.namelist():
                with open(out_file + name[:name.index('.')] + '.json', 'w') as js:
                    file = zip.open(name)
                    pixels = []
                    h, w = 0, 0
                    for line in file.readlines():
                        line = str(line, "utf-8")
                        sz = len(line)
                        content = line[:sz - 2].split(' ')[1:]
                        for byte in content:
                            if byte == '':
                                continue
                            if byte[0] == '?':
                                pixels.append(0)
                            else:
                                pixels.append(int('0x' + byte, 16))
                    sz = len(pixels)
                    w = self.chooseWith(sz / 1024)
                    h = int(sz / w)
                    data = {'name': name, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                    json.dump(data, js)
                    js.write('\n')
                    cnt += 1
                    print("file's id: %d, name = %s" % (cnt, name))


    def extract_image_virusshare(self, zip_file, out_file):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            zip.setpassword(b"infected")
            for name in zip.namelist():
                if isfile(out_file + name + '.json'):
                    continue
                file = zip.open(name)
                content = file.read()
                if len(content) == 0:
                    continue
                two_first_byte = chr(content[0]) + chr(content[1])
                if two_first_byte != "MZ":
                    continue
                pixels = []
                h, w = 0, 0
                for byte in content:
                    pixels.append(byte)
                sz = len(pixels)
                w = self.chooseWith(sz / 1024)
                h = int(sz / w)
                with open(out_file + name + '.json', 'w') as js:
                    data = {'name': name, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                    json.dump(data, js)
                    js.write('\n')
                    cnt += 1
                    print("file's id: %d, name = %s" % (cnt, name))


    def extract_image_PE_zip(self, zip_file, out_file, str):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            for name in zip.namelist():
                if name.replace(str, '') == '':
                    continue
                file = zip.open(name)
                content = file.read()
                two_first_byte = chr(content[0]) + chr(content[1])
                if two_first_byte != "MZ":
                    continue
                pixels = []
                h, w = 0, 0
                for byte in content:
                    pixels.append(byte)
                sz = len(pixels)
                w = self.chooseWith(sz / 1024)
                h = int(sz / w)
                name_ = name.replace(str, '')
                with open(out_file + name_ + '.json', 'w') as js:
                    data = {'name': name_, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                    json.dump(data, js)
                    js.write('\n')
                    cnt += 1
                    print("file's id: %d, name = %s" % (cnt, name_))


    def read_test_data(self, zip_file, out_file, pwd):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            zip.setpassword(pwd.encode())
            for name in zip.namelist():
                if 'Samples/' in name and '.' not in name:
                    name_ = name[name.index('Samples/') + 8:]
                    if name_ != "":
                        file = zip.open(name)
                        content = file.read()
                        pixels = []
                        h, w = 0, 0
                        for byte in content:
                            pixels.append(byte)
                        sz = len(pixels)
                        w = self.chooseWith(sz / 1024)
                        h = int(sz / w)
                        with open(out_file + name_ + '.json', 'w') as js:
                            data = {'name': name_, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                            json.dump(data, js)
                            js.write('\n')
                            cnt += 1
                            print("file's id: %d, name = %s" % (cnt, name))


    def extract_image_PE(self, path, dst):
        cnt = 0
        for f in listdir(path):
            file = join(path, f)
            if isfile(file) == False:
                continue
            pixels, h, w = self.read_PE(file)
            cnt += 1
            with open(dst + f + '.json', 'w') as js:
                data = {'name': f, 'pixels': pixels, 'width': w, 'height': h}
                json.dump(data, js)
            print("file's id: %d, name = %s" % (cnt, f))

        return 0


    def is_byte(self, s):
        if s == '':
            return True
        bytes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        if s[0] in bytes:
            return True
        else:
            return False


    def extract_instructions(self, zip_file, out_file):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            for name in zip.namelist():
                if name.endswith(".asm"):
                    cnt += 1
                    with open(out_file + name[:name.index('.')] + '.json', 'w') as js:
                        file = zip.open(name, 'r')
                        for line in file:
                            ins = []
                            content = [str(x, 'latin') for x in line.split()]
                            if len(content) == 1 or (not content[0].startswith('.text') and not content[0].startswith(
                                    '.code') or ';' in content):
                                continue
                            for x in content[1:]:
                                if self.is_byte(x) == False:
                                    ins.append(x)

                            data = {'instructions': ins, 'name': name}
                            json.dump(data, js)
                            js.write('\n')
                        print("file's id: %d, name = %s" % (cnt, name))


    # def extract_dll(self, zip_file, out_file):
    #     cnt = 0
    #     with zipfile.ZipFile(zip_file) as zip:
    #         for name in zip.namelist():
    #             if name.endswith(".asm"):
    #                 cnt += 1
    #                 with open(out_file + name[:name.index('.')] + '.json', 'w') as js:
    #                     file = zip.open(name, 'r')
    #                     ins = []
    #                     for line in file:
    #                         content = [str(x, 'latin') for x in line.split()]
    #                         if len(content) == 1 or not content[0].startswith('.idata') or 'extrn' not in content:
    #                             continue
    #                         dll_name = content[content.index('extrn') + 1].split(':')[0]
    #                         ins.append(dll_name)
    #                     data = {'dll': ins, 'name': name}
    #                     json.dump(data, js)
    #                     print("file's id: %d, name = %s" % (cnt, name))


    def extract_dll_PE(self, path, dst):
        cnt = 0
        for f in listdir(path):
            file = join(path, f)
            if isfile(file):
                cnt += 1
                dll = self.extract_import_lib(file)
                if len(dll) == 0:
                    continue
                with open(dst + f + '.json', 'w') as js:
                    data = {'name': f, 'dll': dll}
                    json.dump(data, js)
                    print("file's id: %d, name = %s" % (cnt, f))


    def bilinear_interpolation(self, pixels, w, h, w_2, h_2):
        temp = []
        x_ratio = (int)(((w - 1) << 16) / w_2)
        y_ratio = (int)(((h - 1) << 16) / h_2)
        y = 0
        for i in range(h_2):
            yr = (int)(y >> 16)
            y_diff = y - (yr << 16)
            one_min_y_diff = 65536 - y_diff
            y_index = yr * w
            x = 0
            for j in range(w_2):
                xr = (int)(x >> 16)
                x_diff = x - (xr << 16)
                one_min_x_diff = 65536 - x_diff
                index = y_index + xr

                A = pixels[index] & 0xff
                B = pixels[index + 1] & 0xff
                C = pixels[index + w] & 0xff
                D = pixels[index + w + 1] & 0xff

                # Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + D(w)(h)
                gray = (int)((
                                 A * one_min_x_diff * one_min_y_diff + B * x_diff * one_min_y_diff + C * y_diff * one_min_x_diff + D * x_diff * y_diff) >> 32)

                temp.append(gray)
                x += x_ratio
            y += y_ratio;
        return temp


    def image_feature(self, path, f, w_2, h_2, n_chanels=3):
        with open(join(path, f), 'r') as js:
            data = [json.loads(line) for line in js]
            name = data[0]['name']
            pixels = data[0]['pixels']
            w = data[0]['width']
            h = data[0]['height']
            # h_ = int(h / n_chanels)
            # h_r = h % n_chanels
            # pixels_ = []
            # for i in range(n_chanels):
            #     if i != n_chanels - 1:
            #         pixels_.append(self.resize_image(pixels[w*i*h_: w*h_*(i+1)], w, h_, w_2, h_2))
            #     else:
            #         pixels_.append(self.resize_image(pixels[w*i*h_: w*h], w, h_ + h_r, w_2, h_2))

            pixels_rzed = self.resize_image(pixels, w, h, w_2, h_2)
            return pixels_rzed


    def resize_image(self, pixels, w, h, w_2, h_2):
        im = np.reshape(np.array(pixels), (h, w)).astype(np.uint8)
        thumbnail = cv.resize(im, (w_2, h_2), interpolation=cv.INTER_AREA)
        return list(np.reshape(thumbnail, (w_2 * h_2,)).astype(float))


    def dll_feature(self, file, dic):
        with open(file, 'r') as js:
            data = [json.loads(line) for line in js]
            apis = data[0]['feature']
            feature = [0]*(len(dic))
            for api in apis:
                if api in dic:
                    id = dic[api]
                    feature [id] = 1
            return feature

    def build_dll_feature(self, src, dst, dic):
        cnt = 0
        for f in listdir(src):
            if isfile(dst + f):
                continue
            file = join(src, f)
            fea = self.dll_feature(file, dic)
            with open(dst + f + '.json', 'w') as js:
                data = {'name': f, 'feature': fea}
                json.dump(data, js)
                js.close()
                cnt += 1
                print("file's id: %d, name = %s" % (cnt, f))


    def build_image_feature(self, path, dst, w_2, h_2):
        cnt = 0
        for f in listdir(path):
            try:
                if isfile(dst + f):
                    continue
                cnt += 1
                pixels = self.image_feature(path, f, w_2, h_2)
                print(pixels)
                with open(dst + f[:f.index('.')] + '.json', 'w') as js:
                    data = {'name': f[:f.index('.')], 'pixels': pixels, 'width': w_2, 'height': h_2}
                    json.dump(data, js)
                    print("file's id: %d, name = %s" % (cnt, f[:f.index('.')]))
            except Exception:
                print("error")


    def build_ins_feature(self):
        # TO DO
        return 0


    def read_image_json(self, file):
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
            pixels = data[0]['pixels']
            w = data[0]['width']
            h = data[0]['height']
            return pixels, w, h


    def check_type(self, zip_file):
        # cmd = shlex.split('file --mime-type {0}'.format(filename))
        # result = subprocess.check_output(cmd)
        # mime_type = str(result.split()[-1], 'latin')
        # print(mime_type)
        cnt = 0
        dem = 0
        with zipfile.ZipFile(zip_file) as zip:
            zip.setpassword(b"infected")
            for name in zip.namelist():
                dem += 1
                print("file's id = %d" % (dem))
                file = zip.open(name)
                b = str(file.read(2), 'latin')
                if b == "MZ":
                    cnt += 1
        print("number of file exe: %d" % (cnt))


    def saveImage(self, src, dst):
        data, width, height, name = self.read_data(src, 100)
        for (p, w, h, n) in zip(data, width, height, name):
            self.display(p, w, h, n, dst)


ex = Extractor()
# pixels, h, w = ex.read_Binary("D:\\Data\\Malware\\dataSample\\Ig2DB5tSiEy1cJvV0zdw.bytes")
# print(pixels)
# ex.resize_image(pixels, w, h, 64, 64)

# pixels, w, h = ex.read_image_json("D:/Data/Malware/Zip/image_fea/0ACDbR5M3ZhBJajygTuf.json")
# ex.resize_image(pixels, w, h, 64, 64)

# ex.extract_image("D:/Data/Malware/Zip/dataSample.zip", "D:/Data/Malware/Zip/")

# w2 = 64
# h2 = 64
#
# temp = ex.resizeBilinearGray("D:\\Data\\Malware\\dataSample\\0A32eTdBKayjCWhZqDOQ.bytes", w2, h2, True)
# ex.display(temp, w2, h2)

# ex.extract_instructions("D:/Data/Malware/Zip/dataSample.zip", "D:/Data/Malware/Zip/")

# ex.extract_dll("D:/Data/Malware/Zip/dataSample.zip", "D:/Data/Malware/Zip/")

# ex.build_image_feature("D:/Data/Malware/Zip/image/", "D:/Data/Malware/Zip/image_fea/", 64, 64)

# src = sys.argv[1]
# dst = sys.argv[-1]
# ex.extract_to_image(src, dst)
# ex.extract_dll_PE("D:/Data/Malware/clear/benign/", "D:/Data/Malware/clear/benign/import/")
# ex.check_type("D:/VirusShare_00309.zip")
#
# ex.read_test_data("D:/Data/Malware/clear/APT_Group.zip", "D:/Data/Malware/clear/benign/import/")
# print('dangmc'.encode())
# ex.saveImage("D:/Data/Malware/clear/virus/", "D:/Documents/Image/")

# ex.extract_2d_histogram("D:/Data/Malware/clear/image/", "D:/Data/Malware/clear/histogram/", 1024)
src = "D:/Data/Malware/clear/VT/files_16/"
dst = "D:/Data/Malware/clear/VT/image/"
ex.extract_image_file_PE(src, dst)
# cnt_t = 0
# cnt_f = 0
# for f in listdir(path):
#     dll = ex.extract_import_lib(join(path, f))
#     if len(dll) == 0:
#         cnt_f += 1
#     else:
#         cnt_t += 1
#         print(cnt_t)
# ex.extract_image_file_PE(src, dst)
# text_file = open("D:/Data/Malware/clear/VT/api.txt", 'r')
# lines = text_file.read().split(',')
# api = {}
# for i in range(len(lines)):
#     api[lines[i]] = i
# ex.build_dll_feature(src, dst, api)
