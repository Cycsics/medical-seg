import warnings
import os
import pandas as pd
from ultralytics import YOLO, RTDETR

warnings.filterwarnings('ignore')

def calculate_f2_score(precision, recall):
    print(precision, recall)
    return 5 * precision * recall / (4 * precision + recall)

def validate_and_collect_data(model_name, model_path, data_yaml, imgsz, batch):
    print("YOLO")
    models = YOLO(model_path)
    info = models.info(detailed=True, verbose=True)
    # print("INFORMATION:\t",info)
    metrics = models.val(data=data_yaml, split='val', imgsz=imgsz, batch=batch, save_json=True, project='runs/val', name=model_name)
    # print(metrics)
    # print(metrics.box)
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    f2_score = calculate_f2_score(precision, recall)
    # print("Available keys in results_dict:", metrics.results_dict.keys())
    # ap_per_class = metrics.results_dict['metrics/ap_class']
    performance_data = {
        # "model": model_name.split('-')[0],
        "model": model_name,
        "Precision": precision,
        "F1-Score": metrics.box.f1,
        "F2-Score": f2_score,
        "Recall": recall,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "inference": "{:.1f}".format(metrics.speed['inference']) + "ms",  # FPS calculation
        "FPS": 1000 / metrics.speed['inference'],  # FPS calculation
        "FLOPS": "{:.1f}".format(info[3])+"G"
    }
    # class_labels = metrics.results_dict['metrics/class_labels']
    # for class_label, ap in zip(class_labels, ap_per_class):
    #     performance_data[f"AP_{class_label}"] = ap * 100  # Convert to percentage
    print(performance_data)
    return performance_data

if __name__ == '__main__':
    
    # Get model directories from runs/REEF folder
    # base_directory = 'runs/REEF'
    # data_yaml = f'REEF.yaml'
    # imgsz = [1280, 720]    
    # model_dirs = [
    #     "RTDETR-resnet50-SGD-1920-0.6-83",
    #     'yolov6n',
    #     'yolov8n-p2-SGD-1280-0.6-165',
    #     'yolov9c-SGD-1280-0.6-16',
    #     'yolov8n-C2f_iRMB_EMA-HAT-P5-SGD-1280-0.6-166',
    # ]
        
    # base_directory = 'runs/URPC'
    # data_yaml = 'URPC.yaml'
    # imgsz = [1920, 1080]
    # model_dirs = [
    #     'RTDETR-resnet50-SGD-1920-0.6-8'
    #     'yolov6-SGD-1920-0.6-8',
    #     'yolov8n--SGD-1920-0.6-12',
    #     'yolov9c-SGD-1920-0.6-3',
    #     'yolov8n-C2f_iRMB_EMA-HAT-P5-SGD-1920-0.6-122',
    # ]    
    
    base_directory = 'runs/cholecseg8k'
    data_yaml = 'datasets/cholecseg8k_yolo/config.yaml'
    imgsz = 640
    # imgsz = [1920, 1080]
    model_dirs = [
        # 'yolo11-seg_64_SGD_100_',
        # 'yolo11-seg-ASF_64_SGD_100_',
        # 'yolo11-seg-ASF-DSD_64_SGD_100_',
        # 'yolo11-seg-ASF-SDI_64_SGD_100_',
        'yolo11-seg-ASF-SDI-DSD_64_SGD_100_',
    ]
    batch = 64


    

    #     'yolov8--EMA-SGD-1280-0.6-16',
    # #     'yolov5n',
    # #     'yolov6n',
    #     'yolov9c-SGD-1280-0.6-16',
    #     'yolov8n--SGD-1280-0.6-16',
    #     'yolov8n-p6-SGD-1280-0.6-16',
    #     'yolov8n-p2-SGD-1280-0.6-165',
        
        
    # #     # 'yolov8n-HAT-SGD-1280-0.6-162',
    #     'yolov8n-C2f-iEMA-SGD-1280-0.6-162',
    #     'yolov8n-C2f-iRMB-SGD-1280-0.6-162',
    #     # 'yolov8n-C2f_iRMB_EMA-SGD-1280-0.6-162',
    #     'yolov8n-C2f_iRMB_EMA-HAT-P5-SGD-1280-0.6-166',
    # ]
    
    results = []

    for model_dir in model_dirs:
        model_path = os.path.join(base_directory, model_dir, 'train', 'weights', 'best.pt')
        print(model_path)
        if os.path.exists(model_path):
            performance_data = validate_and_collect_data(model_dir, model_path, data_yaml, imgsz, batch)
            results.append(performance_data)
        else:
            print('Not exist')
    # Convert results to Pandas DataFrame
    print(results)
    df = pd.DataFrame(results)
    print(df)
