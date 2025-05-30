from ultralytics import SAM, YOLO
import cv2
import numpy as np
import time
import os

# 加载模型

model_list = [
    # 'yolo11-seg_64_SGD_100_',
    # 'yolo11-seg-ASF_64_SGD_100_',
    'yolo11-seg-ASF-SDI_64_SGD_100_',
    # 'yolo11-seg-ASF-DSD_64_SGD_100_',
    # 'yolo11-seg-ASF-SDI-DSD_64_SGD_100_',
]

for model_name in model_list:
    model = YOLO(f'runs\cholecseg8k\{model_name}/train/weights/best.pt', task='segment')

    # 显示模型信息
    model.info()

    # 图像文件夹路径
    image_folder = 'datasets/cholecseg8k_yolo/raw_images'
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # 确保顺序一致

    # 创建保存目录
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    # 设置延迟时间（毫秒）
    display_time = 10  # 每张图片显示0.5秒

    for image_file in image_files:
        # 读取原始图像
        image_path = os.path.join(image_folder, image_file)
        original_image = cv2.imread(image_path)
        print(image_file)
        if original_image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        # 进行推断
        result = model(original_image)[0]
        
        # 获取分割结果
        segmentation_image = result.plot(conf=True, masks=True, labels=False, boxes=False)
        
        # 调整图像大小
        h, w = original_image.shape[:2]
        segmentation_image = cv2.resize(segmentation_image, (w, h))
        
        # 添加标签
        cv2.putText(original_image, "Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(segmentation_image, "Segmentation", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 水平拼接两张图像
        comparison = np.hstack((original_image, segmentation_image))
        
        # 调整大小以适应屏幕
        if comparison.shape[1] > 1920:
            scale = 1920 / comparison.shape[1]
            new_height = int(comparison.shape[0] * scale)
            comparison = cv2.resize(comparison, (1920, new_height))
        
        # 显示图像
        cv2.imshow('Original vs Segmentation', comparison)
        
        # 自动保存结果
        # save_path = os.path.join(save_dir, f'comparison_{image_file}')
        # cv2.imwrite(save_path, comparison)
        
        # 添加短暂延迟，允许用户查看和按键退出
        key = cv2.waitKey(display_time)
        if key == ord('q'):  # 如果按下'q'，则退出循环
            break

    cv2.destroyAllWindows()
    print(f"所有结果已保存到 {save_dir} 目录")