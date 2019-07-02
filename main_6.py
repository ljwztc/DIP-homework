# coding:utf-8
# describe: 用二值形态学的方法实现边界的提取
import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_binary_img(img):
    img = img <= 128
    img = img.astype(np.uint8)
    return img

def image_erosion(img):
    model = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    long, width = img.shape
    img_erosion = np.zeros([long, width])
    for i in range(long):
        for ii in range(width):
            if i == 0 or i == long - 1 or ii == 0 or ii == width - 1:
                img_erosion[i][ii] = 0
            else:
                store = img[range(i - 1, i + 2), :]
                store = store[:, range(ii - 1, ii + 2)]
                result = model == store
                flag = 1
                for j in range(3):
                    for jj in range(3):
                        flag = flag & result[j][jj]
                if flag:
                    img_erosion[i][ii] = 1
                else:
                    img_erosion[i][ii] = 0
    return img_erosion


def plot_binary_img(img, img_name):
    long, width = img.shape
    img = img.astype(np.uint8)
    for i in range(long):
        for ii in range(width):
            if img[i, ii] == 1:
                img[i, ii] = 255
            else:
                img[i, ii] = 0
    cv2.imshow(img_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

image_path = '/Users/mac/Desktop/test.bmp'
img = cv2.imread(image_path)
print(img.shape)
'''
img = cv2.imread(image_path, 0)
img = convert_to_binary_img(img)
img_erosion = image_erosion(img)
plot_binary_img(img, "binary_img")
plot_binary_img(img_erosion, "erosion_img")
img_edge = img - img_erosion
plot_binary_img(img_edge, "img_edge")
'''
