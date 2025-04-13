import os
import cv2
import numpy as np

def extract_bordered_region(img, padding_ratio=0.05):
    """
    提取黑色边框内的文字区域
    padding_ratio: 裁剪时保留的边框内边距比例（避免残留边框）
    """

    # cv2.namedWindow('Display Window', cv2.WINDOW_NORMAL)

    # 转换为灰度图并增强对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # blurred = gray
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(24, 24))
    enhanced = clahe.apply(blurred)

    # 检测黑色边框（使用自适应阈值）
    binary = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 61, 16)

    # 形态学操作强化边框
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

        height, width = gray.shape[:2]
        # 提取轮廓点的x坐标
        x_coords = cnt[:, 0, 0]  # 轮廓点坐标格式为 (n,1,2)，提取x列
        # 检查是否接触左侧（x=0）和右侧（x=width-1）
        touches_left = np.any(x_coords <= 1)
        touches_right = np.any(x_coords >= width - 2)
        # 若同时接触左右两侧，则跳过；否则保留
        if touches_left and touches_right:
            continue

        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            selected_contour = cnt
            approx_final = approx  # 保存近似结果供后续使用
            break  # 找到符合条件的即跳出循环

    # 显示框选区域
    # result = img.copy()
    # cv2.drawContours(result, selected_contour, -1, (0, 0, 255), 10)
    # cv2.imshow('Display Window', result)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Display Window')

    if selected_contour is None:
        raise ValueError("所有候选轮廓均不符合四边形要求")


    # 透视变换校正
    # src_points = order_points(approx.reshape(4, 2))
    src_points = order_points(approx_final.reshape(4, 2))
    w, h = 400, 200  # 输出尺寸根据实际需求调整
    dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (w, h))

    # 裁剪内部区域（排除边框）
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    cropped = warped[pad_h:h - pad_h, pad_w:w - pad_w]

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

    参数：
    img : numpy.ndarray
        OpenCV图像对象，建议为单通道二值图像
    strict_mode : bool (默认True)
        True: 严格检查输入是否为二值图像
        False: 自动转换非二值图像
    return_copy : bool (默认True)
        True: 返回修改后的副本，保留原始图像
        False: 直接修改原始图像

    返回：
    numpy.ndarray : 处理后的图像

    异常：
    ValueError: 当strict_mode=True且检测到非二值图像时抛出
    """
    # 输入验证
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须是numpy数组")

    # 自动转换彩色图像为灰度
    if len(img.shape) == 3:
        if strict_mode:
            raise ValueError("严格模式下不接受彩色图像输入")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 严格模式验证二值图像
    if strict_mode:
        unique_vals = np.unique(img)
        if len(unique_vals) > 2 or (set(unique_vals) - {0, 255}):
            raise ValueError("严格模式需要标准二值图像(0/255)")

    # 创建副本或直接操作
    target = img.copy() if return_copy else img

    # 计算切割位置
    height = target.shape[0]
    cutoff = int(height * 0.5)

    # 应用遮罩
    if cutoff > 0:  # 防止高度过小的情况
        target[:cutoff, :] = 0

    return target

