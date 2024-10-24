import cv2
import torch
from ultralytics import YOLO
import os

captureResolution = 1280 # Stream resolution for processing;

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, captureResolution)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, captureResolution)

# ******CONFIG*******
useSearch = False

civilMode = False
militaryModelToUse = 0; # 0 - Large model affects performance; 1 - Optimized model with less accuracy;
# *******************

model = None

# example query to search
query = "person"

if not useSearch:
        if civilMode:     
            # use YOLO pre-trained model for general object detection including people, animals, props, etc.;
            # yolov8n.pt - optimal FPS;
            # yolov10n.pt - best accuracy;
            model = YOLO("yolov8n.pt")
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_path, "CustomModels", "Vehicles_L", "RussianVehicles_L.pt") if militaryModelToUse == 0 else os.path.join(base_path, "CustomModels", "Vehicles_S", "RussianVehicles_S.pt")
            # Load model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
else:
        # use model by searching query in YOLO-world dataset for general objects detection;
        model = YOLO("yolov8s-worldv2.pt")
        model.set_classes([query])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    if civilMode:
        #print(results)
        annotated_frame = results[0].plot()
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time
        text = f'FPS: {fps:.1f}'

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10

        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Camera", annotated_frame)
    else:
        # only printing results for now
        # //TODO: implement drawing bounding box like in civilMode;
        print(results.pandas().xyxy[0])
        cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
