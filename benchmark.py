from ultralytics.utils.benchmarks import benchmark, ProfileModels

# Benchmark on GPU
ProfileModels('runs/REEF/yolov8n--Adam-1280-0.6-162/weights/', imgsz=640, trt=False).profile()
# benchmark(model='runs/REEF/yolov8n--Adam-1280-0.6-162/weights/best.pt', data='URPC.yaml', imgsz=640, half=False, device=0)
# print()