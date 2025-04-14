import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR, draw_ocr

# 设置Tesseract路径
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_bordered_region(img, card_size=(1200, 600)):
    """
    提取黑色边框内的文字区域
    padding_ratio: 裁剪时保留的边框内边距比例（避免残留边框）
    """

    # cv2.namedWindow('Display Window', cv2.WINDOW_NORMAL)

    # 转换为灰度图并增强对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(24, 24))
    enhanced = clahe.apply(blurred)

    # 检测黑色边框（使用自适应阈值）
    binary = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 61, 16)
    # 形态学强化边框
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 查找所有轮廓
    # inverted = cv2.bitwise_not(closed)
    # inverted = mask_upper(inverted)
    inverted = mask_upper(closed)
    inverted = cv2.bitwise_not(inverted)

    # 显示图像预处理过程
    # cv2.imshow("Display Window",blurred)
    # cv2.waitKey(0)
    # cv2.imshow("Display Window",enhanced)
    # cv2.waitKey(0)
    # cv2.imshow("Display Window",binary)
    # cv2.waitKey(0)
    # cv2.imshow("Display Window",inverted)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("未检测到任何轮廓")

    # 获取图像总面积（假设closed是二值图，与原图尺寸一致）
    img_height, img_width = closed.shape[:2]
    total_area = img_height * img_width

    # 过滤掉面积超过图像50%的轮廓
    filtered_contours = [
        cnt for cnt in contours
        if 0.4 * total_area >= cv2.contourArea(cnt) > 0.01 * total_area  # 只保留面积合理的轮廓
    ]

    if not filtered_contours:
        raise ValueError("检测到的轮廓面积均不符合要求，请检查拍摄条件")

    # 将过滤后的轮廓按面积从大到小排序
    sorted_contours = sorted(filtered_contours,
                             key=cv2.contourArea,
                             reverse=True)

    # 遍历寻找第一个四边形轮廓
    selected_contour = None
    approx_final = None
    for cnt in sorted_contours:
        # result = img.copy()
        # cv2.drawContours(result, cnt, -1, (0, 0, 255), 10)
        # cv2.imshow('Display Window', result)
        # cv2.waitKey(0)

        # 删除横跨的误判
        height, width = gray.shape[:2]
        x_coords = cnt[:, 0, 0]
        touches_left = np.any(x_coords <= 1)
        touches_right = np.any(x_coords >= width - 2)
        if touches_left and touches_right:
            continue

        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            selected_contour = cnt
            approx_final = approx
            break

            # 显示框选区域
    # result = img.copy()
    # cv2.drawContours(result, selected_contour, -1, (0, 0, 255), 10)
    # cv2.imshow('Display Window', result)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Display Window')

    if selected_contour is None:
        raise ValueError("所有候选轮廓均不符合四边形要求")

    # 透视变换校正
    src_points = order_points(approx_final.reshape(4, 2))
    w, h = card_size
    dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, card_size)

    # 裁剪内部区域（排除边框）
    padding_ratio = 0.005
    # padding_ratio = 0
    if padding_ratio != 0:
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        cropped = warped[pad_h:h - pad_h, pad_w:w - pad_w]
    else:
        cropped = warped

    return cropped


def order_points(pts):
    """将四个点按左上、右上、右下、左下排序"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect


def mask_upper(img, strict_mode=True, return_copy=False):
    """
    将二值图像的上半部分40%区域置为黑色(0)
    img : numpy.ndarray 二值图像
    strict_mode : bool 是否检查输入为二值图像
    return_copy : bool
        True: 返回修改后的副本，保留原始图像
        False: 直接修改原始图像
    """
    # 输入验证
    if not isinstance(img, np.ndarray):
        raise TypeError("输入不是是numpy数组")

    # 自动转换彩色图像为灰度
    if len(img.shape) == 3:
        if strict_mode:
            raise ValueError("不是二值图像")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 严格模式验证二值图像
    if strict_mode:
        unique_vals = np.unique(img)
        if len(unique_vals) > 2 or (set(unique_vals) - {0, 255}):
            raise ValueError("不是二值图像")

    # 创建副本或直接操作
    target = img.copy() if return_copy else img

    # 计算切割位置
    height = target.shape[0]
    cutoff = int(height * 0.5)

    if cutoff > 0:  # 防止高度过小的情况
        target[:cutoff, :] = 0

    return target


def single_card_OCR(img, first_line=True):
    """
    调用 pytesseract 识别文字
    输入图像 img, first_line=True 只取出第一行
    返回 string
    """

    ocr = PaddleOCR(lang='ch')
    result = ocr.ocr(img)
    text = None
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = line[1][0]
            if '壶' in text:
                # print(text)
                break
        if text != None:
            break
    # text = result[0][0][1][0]
    # print(text)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # 预处理
    # h, w = img.shape[:2]
    # print(h, w)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(w // 10, h // 10))
    # enhanced = clahe.apply(gray)
    #
    # blockSize = (int(h / 4) & 1) + int(h / 4) + 1
    # # print(blockSize)
    # binary = cv2.adaptiveThreshold(enhanced, 255,
    #                                cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                cv2.THRESH_BINARY, blockSize, 35)
    #
    # # # 定义ROI参数 (x,y)为左上角坐标，(a,b)为宽度和高度
    # # x, y = 20, 100
    # # dx, dy = 110, 300
    # # title = np.zeros((h, w), dtype=np.uint8)
    # # title = cv2.bitwise_not(title)
    # # title[200:200+dx, 100:100+dy] = binary[x:x+dx, y:y+dy]
    # #
    # new_w = 1000
    # title = cv2.resize(binary,(new_w,new_w//2))
    #
    # # roi = enhanced[20:130, 100:400]
    #
    # # cv2.imshow("roi", roi)
    # # cv2.waitKey(0)
    #
    # text = pytesseract.image_to_string(title, lang='chi_sim')
    # print(text)
    # cv2.waitKey(0)
    # first_line = text.splitlines()[0]  # 直接获取第一行
    # cha_hu_name = first_line.split(' ')[-1]

    return text.strip()


def ocr_2(img):
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
    # to switch the language model in order

    h, w = img.shape[:2]
    img = img[int(h * 0.4):h, :]
    h = int(h * 0.6)
    # cv2.imshow("b_img",cv2.resize(img,(int(400*w/h),400)))
    # cv2.waitKey(0)
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(img, cls=True)
    text = None
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = line[1][0]
            conf = line[1][1]
            if ('壶' in text) and (conf > 0.6):
                # print(text)
                break
        if text is not None:
            break

    if text is None:
        raise ValueError("无法确认有壶的名称")
    elif not ('\u4e00' <= text[0] <= '\u9fff'):
        return text[1:]

    return text.strip()
