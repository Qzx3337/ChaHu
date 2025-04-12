import cv2
import pytesseract
import numpy as np
from PIL import Image


# 设置Tesseract路径（Windows需要）
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

def locate_text_region(image_path):

    # cv2.namedWindow('Display Window', cv2.WINDOW_NORMAL)

    # 读取图片
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 定义下半部分ROI（从图片中间到底部）
    roi_height_ratio = 0.5  # 使用下半50%区域
    roi_start_y = int(h * (1 - roi_height_ratio))
    roi = img[roi_start_y:h, 0:w]  # 横向覆盖整个宽度

    # 灰度化 + 降噪
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 25)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 形态学操作（膨胀连接文字区域）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(dilated, kernel, iterations=10)

    # cv2.imshow("Display Window",dilated)
    # cv2.waitKey(0)

    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓（面积和宽高比）
    text_contours = []
    for cnt in contours:
        # result = roi.copy()
        # cv2.drawContours(result, cnt, -1, (0, 0, 255), 10)
        # cv2.imshow('Display Window', result)
        # cv2.waitKey(0)

        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch  # 直接使用矩形面积更高效

        # 筛选条件（可根据实际调整）
        if area > 100000 and ch > 40:  # 最小高度20像素，面积>1000
            print(area)
            text_contours.append(cnt)

    # 合并所有符合条件的轮廓
    if text_contours:
        combined = np.vstack(text_contours)
        x, y, cw, ch = cv2.boundingRect(combined)

        # 坐标转换到原图
        x_abs = x
        y_abs = y + roi_start_y  # 关键坐标转换
        return (x_abs, y_abs, cw, ch)
    else:
        return None


def extract_text(image_path):
    region = locate_text_region(image_path)
    if not region:
        return "未检测到文字区域"

    cv2.namedWindow('Display Window', cv2.WINDOW_NORMAL)

    # 裁剪区域
    img = cv2.imread(image_path)
    x, y, w, h = region
    cropped = img[y:y + h, x:x + w]

    # 高级预处理
    # processed = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # processed = cv2.medianBlur(processed, 3)
    # processed = cv2.threshold(processed, 0, 255,
    #                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imshow('Display Window', processed)
    cv2.waitKey(0)

    # OCR识别（中英文混合）
    text = pytesseract.image_to_string(processed, lang='chi_sim')
    return text.strip()


if __name__ == "__main__":
    image_path = "IMG_20240628_140547.jpg"
    result = extract_text(image_path)
    print("识别结果：\n", "=" * 30, "\n", result)