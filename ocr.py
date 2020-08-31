import re
from random import random

import cv2
import numpy as np

from paddleocr import PaddleOCR


# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


# 适配原四边形点集
def adaPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box/pro
    box_pro = np.trunc(box_pro)
    return box_pro


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')
    transform = cv2.getPerspectiveTransform(box, dst_rect)
    return cv2.warpPerspective(image, transform, (w, h))


# 固定尺寸
def resize(image, width=1000):
    h, w, _ = image.shape
    if w > width:
        image = cv2.resize(image, (width, int(width * h / w)))
    return image


def is_front(result):
    found = False
    front = False

    for line in result:
        if line[1][0] in ('中华人民共和国', '公民身份号码'):
            found = True
            front = True if line[1][0] == '公民身份号码' else False
            break

    return found, front


def identity(image, ocr):
    """
    paddleOcr 识别身份证，如果识别不出来则旋转180度尝试
    :param image:
    :param ocr:
    :return:
    """
    result = ocr.ocr(image)
    found, front = is_front(result)

    if found is False:
        image = np.rot90(image, 2)
        result = ocr.ocr(image)
        found, front = is_front(result)

    h, w, _ = image.shape

    # 根据位置定位需要的文字
    if front:
        ret = {'name': '', 'gender': '', 'nation': '', 'birthday': '', 'card': '', 'address': ''}
    else:
        ret = {'effective_date': '', 'expire_date': ''}

    if found:
        if front:
            for line in result:
                if line[1][1] < 0.7:
                    # 置信度小于0.7不处理
                    continue

                y = int(line[0][0][1])
                if y < int(h * 0.2028):
                    ret['name'] = line[1][0].replace('姓名', '')
                elif y < int(h * 0.3317):
                    if '性别' in line[1][0] and '民族' in line[1][0]:
                        ret['gender'] = line[1][0][2:3]
                        ret['nation'] = line[1][0][5:6]
                    else:
                        if '性别' in line[1][0]:
                            ret['gender'] = line[1][0].replace('性别', '')
                        elif '民族' in line[1][0]:
                            ret['nation'] = line[1][0].replace('民族', '')
                elif y < int(h * 0.4431):
                    ret['birthday'] = ret['birthday'] + line[1][0].replace('出生', '')
                elif y < int(h * 0.7616):
                    ret['address'] = ret['address'] + line[1][0].replace('住址', '')
                else:
                    if re.search('^\d{17}[\dXx]{1}$', line[1][0]) is not None:
                        ret['card'] = line[1][0]
        else:
            for line in result:
                if line[1][1] < 0.7:
                    # 置信度小于0.7不处理
                    continue

                y = int(line[0][0][1])
                if y > int(h * 0.7976) and line[0][0][0] > int(w * 0.3862):
                    date = [item.strip() for item in line[1][0].split('-')]
                    ret['effective_date'] = date[0]
                    ret['expire_date'] = date[1]

    return ret


def main(src, ocr):
    im = cv2.imread(src)
    im = resize(im, 2000)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)     # 转化为灰度图
    blur = cv2.GaussianBlur(gray, (3, 3), 0)        # 用高斯滤波处理原图像降噪
    canny = cv2.Canny(blur, 20, 30)                 # 20是最小阈值,50是最大阈值 边缘检测
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(canny, kernel, iterations=1)              # 膨胀一下，来连接边缘
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 找边框

    ret = {'name': '', 'gender': '', 'nation': '', 'birthday': '', 'card': '', 'address': '', 'effective_date': '',
           'expire_date': ''}
    for i in range(len(contours)):
        contour = contours[i]
        if cv2.contourArea(contour) > 400000:
            boxes = getBoxPoint(contour)
            boxes = adaPoint(boxes, 1.0)
            boxes = orderPoints(boxes)
            # 透视变化
            warped = warpImage(im, boxes)
            ret = {**ret, **identity(warped, ocr)}

    return ret


if __name__ == '__main__':
    paddle = PaddleOCR(det_model_dir='./interface/ch_det_mv3_db/', rec_model_dir='./interface/ch_rec_mv3_crnn/', use_gpu=False)

    print(main('/Users/liwd/Downloads/123.jpeg', paddle))
    print(main('/Users/liwd/Downloads/WechatIMG22.jpeg', paddle))
