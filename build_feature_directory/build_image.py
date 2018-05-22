import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import random_projection
import pefile, zipfile, json
import sys
import cv2 as cv

from os import listdir
from os.path import isfile, join


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
    def display(self, pixels, w, h, n, path):
        g = np.reshape(pixels, (h, w))
        plt.imshow(g, cmap=cm.gray)
        plt.savefig(path + n + '.png')
        print("saved " + n)
        # plt.show()   

    def saveImage(self, src, dst):
        data, width, height, name = self.read_data(src, 100)
        for (p, w, h, n) in zip(data, width, height, name):
            self.display(p, w, h, n, dst)
       
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


    def extract_import_lib(self, file):
        pe = pefile.PE(file)
        pe.parse_data_directories()
        imps = []
        try:
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name == None:
                        continue
                    imps.append(str(imp.name, "latin"))
        except Exception:
            print("don't have imported libraries")
        return imps

    def extract_to_image(self, zip_file, out_file):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            for name in zip.namelist():
                if name.endswith(".bytes"):

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
                            if len(content) == 1 or (not content[0].startswith('.text') and not content[0].startswith('.code') or ';' in content):
                                continue
                            for x in content[1:]:
                               if self.is_byte(x) == False:
                                  ins.append(x)

                            data = {'instructions': ins, 'name': name}
                            json.dump(data, js)
                            js.write('\n')
                        print("file's id: %d, name = %s" % (cnt, name))

    def extract_dll(self, zip_file, out_file):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            for name in zip.namelist():
                if name.endswith(".asm"):
                    cnt += 1
                    with open(out_file + name[:name.index('.')] + '.json', 'w') as js:
                        file = zip.open(name, 'r')
                        ins = []
                        for line in file:
                            content = [str(x, 'latin') for x in line.split()]
                            if len(content) == 1 or not content[0].startswith('.idata') or 'extrn' not in content:
                                continue
                            dll_name = content[content.index('extrn') + 1].split(':')[0]
                            ins.append(dll_name)
                        data = {'dll': ins, 'name': name}
                        json.dump(data, js)
                        js.write('\n')
                        print("file's id: %d, name = %s" % (cnt, name)) 

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

    def extract_image_PE(self, path, dst):
        cnt = 0
        for f in listdir(path):
            file = join(path, f)
            if isfile(file)==False:
                continue
            pixels, h, w = self.read_PE(file)
            cnt += 1
            with open(dst + f + '.json', 'w') as js:
                data = {'name': f, 'pixels': pixels, 'width': w, 'height': h}
                json.dump(data, js)
            print("file's id: %d, name = %s" % (cnt, f))

        return 0
    
    def image_feature(self, path, f, w_2, h_2):
        with open(join(path, f), 'r') as js:
            data = [json.loads(line) for line in js]
            name = data[0]['name']
            pixels = data[0]['pixels']
            w = data[0]['width']
            h = data[0]['height']
            pixels_rzed = self.resize_image(pixels, w, h, w_2, h_2)
            return pixels_rzed

    def resize_image(self, pixels, w, h, w_2, h_2):

        im = np.reshape(np.array(pixels), (h, w)).astype(np.uint8)
        thumbnail = cv.resize(im, (w_2, h_2), interpolation=cv.INTER_AREA)
        return list(np.reshape(thumbnail, (w_2 * h_2,)).astype(float))

    def build_image_feature(self, path, dst, w_2, h_2):
        cnt = 0
        for f in listdir(path):
            try:
                if isfile(dst + f):
                    print("file exist")
                    continue
                cnt += 1
                pixels = self.image_feature(path, f, w_2, h_2)
                #print(pixels)
                with open(dst + f, 'w') as js:
                    data = {'name': f, 'pixels': pixels, 'width': w_2, 'height': h_2}
                    json.dump(data, js)
                    print("file's id: %d, name = %s" % (cnt, f))
            except Exception:
                print("error")
    def extract_image_virusshare(self, zip_file, out_file):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            zip.setpassword(b"infected")
            for name in zip.namelist():
                if isfile(out_file + name + '.json'):
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
                with open(out_file + name + '.json', 'w') as js:
                    data = {'name': name, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                    json.dump(data, js)
                    js.write('\n')
                    cnt += 1
                    print("file's id: %d, name = %s" % (cnt, name))    
	
    
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

    def extract_image_PE_zip(self, zip_file, out_file, str):
        cnt = 0
        with zipfile.ZipFile(zip_file) as zip:
            for name in zip.namelist():
                file = zip.open(name)
                content = file.read()
                if len(content)==0:
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
                name_ = name.replace(str, '')
                with open(out_file + name_ + '.json', 'w') as js:
                    data = {'name': name_, 'pixels': pixels[0: h * w], 'width': w, 'height': h}
                    json.dump(data, js)
                    js.write('\n')
                    cnt += 1
                    print("file's id: %d, name = %s" % (cnt, name_))
ex = Extractor()
src = sys.argv[1]
dst = sys.argv[-1]
ex.build_image_feature(src,dst, 64, 64)
#ex.extract_image_virusshare(src, dst)
