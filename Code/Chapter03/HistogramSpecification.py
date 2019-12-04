import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib


#  定义函数，计算直方图累积概率
#  计算直方图和累计概率CDF可以直接调用numpy中的函数，我会在此处进行改写尝试
def histCalculate(src):
    row, col = np.shape(src)
    hist = np.zeros(256, dtype=np.float32)  # 直方图
    cumhist = np.zeros(256, dtype=np.float32)  # 累积直方图
    cumProbhist = np.zeros(256, dtype=np.float32)  # 累积概率probability直方图，即Y轴归一化
    for i in range(row):
        for j in range(col):
            hist[src[i][j]] += 1

    cumhist[0] = hist[0]
    for i in range(1, 256):
        cumhist[i] = cumhist[i-1] + hist[i]
    cumProbhist = cumhist/(row*col)
    return cumProbhist

#利用numpy函数计算PDF及CDF
def histCalculate_numpy(src):
    intensity_count = [0] * 256
    height,width = src.shape[:2]
    MN = height*width
    for i in range(0,height):
        for j in range(0,width):
            intensity_count[src[i][j]] += 1

    L = 256
    intensity_count,total_values_used = np.histogram(src.flatten(),L,[0,L])
    PDF_list = intensity_count*(L-1)/src.size
    CDF_list = PDF_list.cumsum()
    return CDF_list


# 定义函数，直方图规定化
def histSpecification(specImg, refeImg):  # specification image and reference image
    spechist = histCalculate_numpy(specImg)  # 计算待匹配直方图
    refehist = histCalculate_numpy(refeImg)  # 计算参考直方图
    corspdValue = np.zeros(256, dtype=np.uint8)  # correspond value
    # 直方图规定化
    for i in range(256):
        diff = np.abs(spechist[i] - refehist[i])
        matchValue = i
        for j in range(256):
            if np.abs(spechist[i] - refehist[j]) < diff:
                diff = np.abs(spechist[i] - refehist[j])
                matchValue = j
        corspdValue[i] = matchValue
    outputImg = cv2.LUT(specImg, corspdValue)
    return outputImg


img = cv2.imread('./Picture/Fig0208(a).tif', cv2.IMREAD_GRAYSCALE)
# 读入参考图像
img1 = cv2.imread('./Picture/Lena.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('input', img)
cv2.imshow('refeImg', img1)
imgOutput = histSpecification(img, img1)
cv2.imshow('output', imgOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()

fig = plt.figure('整个过程直方图显示', (8, 8))
matplotlib.rcParams['font.family'] = 'SimHei'
plt.subplot(311)
plt.plot(histCalculate(img), 'r', lw=1, label='待匹配累积概率直方图')
plt.legend(loc='best')
plt.subplot(312)
plt.plot(histCalculate(img1), 'b', lw=1, label='参考累积概率直方图')
plt.legend(loc='best')
plt.subplot(313)
plt.plot(histCalculate(imgOutput), 'g', lw=1, label='规则化后的概率直方图')
plt.legend(loc='best')

plt.show()