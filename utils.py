import time
import cv2
import os

def h_split(img_path):
    h_path = os.path.join(img_path,'hue')
    if not os.path.exists(h_path):
        os.makedirs(h_path)
    rgb_path = os.listdir(os.path.join(img_path,'input'))
    for i in range(len(rgb_path)):
        h_route = os.path.join(h_path, rgb_path[i])
        rgb_route = os.path.join(img_path,'input',rgb_path[i])
        rgb = cv2.imread(rgb_route)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, _, _ = cv2.split(hsv)
        h = cv2.imwrite(h_route, h)
    return h

def bilnear_h(img_path):
    h_path = os.path.join(img_path,'bilear_hue')
    if not os.path.exists(h_path):
        os.makedirs(h_path)
    rgb_path = os.listdir(os.path.join(img_path,'input'))
    for i in range(len(rgb_path)):
        h_route = os.path.join(h_path, rgb_path[i])
        rgb_route = os.path.join(img_path,'input',rgb_path[i])
        rgb = cv2.imread(rgb_route)
        filtered_image1 = cv2.bilateralFilter(rgb, 5, 75, 75)
        hsv = cv2.cvtColor(filtered_image1, cv2.COLOR_RGB2HSV)
        h, _, _ = cv2.split(hsv)
        h = cv2.imwrite(h_route, h)
    return h

# input_file = './datasets/real1000'
# bilnear_h(input_file)

def s_split(img_path):
    s_path = os.path.join(img_path,'saturation')
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    rgb_path = os.listdir(os.path.join(img_path,'input'))
    for i in range(len(rgb_path)):
        s_route = os.path.join(s_path, rgb_path[i])
        rgb_route = os.path.join(img_path,'input',rgb_path[i])
        rgb = cv2.imread(rgb_route)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)
        s = cv2.imwrite(s_route, s)
    return s

def l_split(img_path):
    l_path = os.path.join(img_path,'luminance')
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    rgb_path = os.listdir(os.path.join(img_path,'input'))
    for i in range(len(rgb_path)):
        l_route = os.path.join(l_path, rgb_path[i])
        rgb_route = os.path.join(img_path,'input',rgb_path[i])
        rgb = cv2.imread(rgb_route)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, _, _ = cv2.split(lab)
        l = cv2.imwrite(l_route, L)
    return l



class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr






