from ultralytics import YOLO
import os 


if __name__ ==  '__main__':
    os.environ['WANDB_DISABLED'] = 'true'


    model = YOLO("yolov8x.pt")
    model.to('cuda')
    model.train(data="smart_kitchen.yaml", epochs=150, batch = 4, patience = 50 )

    metrics = model.val()
    print("------------------")
    print(metrics)

