from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    print(torch.__version__)
    # Load a model
    # model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
    # model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
    # model_name = "yolo11n.pt"
    # model = YOLO("yolo11n-seg.yaml", task='segment').load(model_name)  # build from YAML and transfer weights
    # model_name = 'yolo11-SegmentHead'
    model_name_list = [
        # 'yolo11n-seg',    
        # 'yolo11-SegmentHead',
        # 'yolo11-seg-ASF',
        # 'yolo11-seg-SDI',
        # 'yolo11-seg-ASF-DSD',
        # 'yolo11-seg-ASF-SDI',
        # 'yolo11-seg-ASF-SDI-DSD',
        'yolo11-seg-temporal',
        ]
    for model_name in model_name_list:
        model = YOLO(f'{model_name}.yaml', task='segment') # 续训yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
        # model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights
        batch_size = 64
        optimizer = 'SGD'
        # iou = 0.8 # using SGD
        epochs = 100
        # Train the model
        op_path='datasets\cholecseg8k_yolo'
        config_path = os.path.join(os.getcwd(), op_path, r'config.yaml')
        # Training the model for 30 epochs; here degrees, shear and perspective are augmentation arguments
        model.train(data=config_path,
                project=f'runs/cholecseg8k/{model_name}_{batch_size}_{optimizer}_{epochs}_',
                # task='segment',
                # name=f"{model_size.split('.')[0]}-{(model_name.split('.')[0]).split('yolov8')[1][1:]}-{optimizer}-640-{iou}-{batch_size}",
                cache=False,
                imgsz=640,
                # imgsz=[640, 360],
                # imgsz=[1280, 720],
                # imgsz=[1920, 1080],
                epochs=epochs,
                batch=batch_size,
                close_mosaic=0,
                workers=0,
                device=0,
                optimizer=optimizer, # using SGD
                amp=True,# close amp
                single_cls=False,
                deterministic=True,
                verbose=False,
                # cache=True,
                patience=100,
                # iou=iou
                )
# nohup python med_seg_train_cholecseg8k.py > ./output/"$(date +output_%Y%m%d_%H%M%S)_${$}.log" &
# start /B python med_seg_train_cholecseg8k.py > .\output\output_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
