import os
import pandas as pd
from test_words import *


def operate_all():
    # 配置输入路径
    input_dir = "zisha teapot dataset"
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg'))]
    output_file = "ocr_results.csv"    # 输出文件名

    # 收集结果
    results = []

    # cv2.namedWindow('Processing Result', cv2.WINDOW_NORMAL)

    # 遍历所有图片
    for idx, filename in enumerate(img_files):
        print(f"\n正在处理第 {idx + 1}/{len(img_files)} 张图片：{filename}")
        record = {"file_name": filename, "hu_name": "", "ocr_state": "fail"}

        input_path = os.path.join(input_dir, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"  无法读取图片")
            continue

        try:
            hu_name = ocr_2(img)
            print(hu_name)
            record.update({
                "hu_name": hu_name,
                "ocr_state": "succeed"
            })
        except Exception as e1:
            print(f"  ocr2全局识别失败：{str(e1)}")
            try:
                card = extract_bordered_region(img)
                # cv2.imshow('Processing Result', card)
                # cv2.waitKey(0)
                hu_name = single_card_OCR(card)
                record.update({
                    "hu_name": hu_name,
                    "ocr_state": "succeed"
                })

            except Exception as e2:
                error_msg = f"  ocr1局部处理失败：{str(e2)}"
                print(error_msg)
                record["hu_name"] = f"ocr fail（{error_msg}）"  # 记录具体错误
        finally:
            results.append(record)
    print("\n处理完成！")


    # 创建DataFrame并保存
    if results:
        df = pd.DataFrame(results)
        # 调整列顺序
        df = df[["file_name", "hu_name", "ocr_state"]]
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果已保存到 {output_file}")
        print(f"成功识别：{len(df[df['ocr_state'] == 'succeed'])} 条")
        print(f"识别失败：{len(df[df['ocr_state'] == 'fail'])} 条")
    else:
        print("未找到可处理的图片文件")


def operate_test():
    # 读取图像
    img = cv2.imread("zisha teapot dataset/IMG_20240628_140756.jpg")
    try:
        hu_name = ocr_2(img)
        print(hu_name)
    except Exception as e1:
        print(f"ocr2全局识别失败：{str(e1)}")
        try:
            # 提取框内区域
            text_region = extract_bordered_region(img)

            # 可视化结果
            # cv2.imshow("Original", img)
            # cv2.imshow("Text Region", text_region)
            # cv2.waitKey(0)

            hu_name = single_card_OCR(text_region)
            print(hu_name)
            cv2.waitKey(0)

        except Exception as e:
            print(f"处理失败：{str(e)}")


# operate_test()
operate_all()
