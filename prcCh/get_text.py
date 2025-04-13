from test_words import *

def operate_all():
    # 配置输入路径
    input_dir = "zisha teapot dataset"
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg'))]

    # 设置显示窗口
    # cv2.namedWindow('Processing Result', cv2.WINDOW_NORMAL)

    cnt_success = 0

    # 遍历处理所有图片
    for idx, filename in enumerate(img_files):
        print(f"\n正在处理第 {idx + 1}/{len(img_files)} 张图片：{filename}")

        input_path = os.path.join(input_dir, filename)
        img = cv2.imread(input_path)

        if img is None:
            print(f"  × 无法读取图片")
            continue

        try:
            card = extract_bordered_region(img)
            cv2.imshow('Processing Result', card)
            cv2.waitKey(0)
            cnt_success = cnt_success + 1
            print("  √ 处理成功 - 按任意键继续，ESC键退出")

        except Exception as e:
            print(f"  × 处理失败：{str(e)}")

        # 清除当前显示
        # cv2.destroyAllWindows()

    print(f"success = {cnt_success}")
    print("\n处理完成！")

def operate_test():
    # 读取图像
    img = cv2.imread("zisha teapot dataset/IMG_20240628_142959.jpg")

    try:
        # 提取框内区域
        text_region = extract_bordered_region(img)

        # 可视化结果
        # cv2.imshow("Original", img)
        cv2.imshow("Text Region", text_region)
        cv2.waitKey(0)

    except Exception as e:
        print(f"处理失败：{str(e)}")

operate_test()
# operate_all()

