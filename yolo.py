from ultralytics import YOLO
import numpy as np
import cv2
import time

CONF_THRESH = 0.4

def yolo():
    model = YOLO("yolov8n.pt")
    cap = initialize_video_capture()

    ret, frame = cap.read()
    
    results = model.predict([frame], show=False, conf=CONF_THRESH)
    names = model.names

    name_array = extract_names(results, names)
    box_coordinates = extract_box_coordinates(results)

    items = create_items(name_array, box_coordinates)

    annotated_frame = results[0].plot()
    image = np.array(annotated_frame)
    draw(image, cap, items)

    my_list = [{'name': item['name'], 'pos': item['pos']} for item in items]
    print(my_list)

    # cv2.imshow("Display", image)

    # time.sleep(2)

    clean_up(cap)
    return my_list

def initialize_video_capture():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera
    return cap

def extract_names(results, names):
    name_array = [names[int(c)] for r in results for c in r.boxes.cls]
    return name_array

def extract_box_coordinates(results):
    box_coordinates = [[int(coord) for coord in box.xyxy[0]] for r in results for box in r.boxes]
    return box_coordinates

def create_items(name_array, box_coordinates):
    items = [{"name": name, "x1": x1, "x2": x2, "y1": y1, "y2": y2} 
             for name, (x1, y1, x2, y2) in zip(name_array, box_coordinates)]
    return items

def draw(image, cap, items):
    # get camera resolution
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.line(image, (cam_width//4, 0), (cam_width//4, cam_height), (255, 0, 0), 5)
    cv2.line(image, (cam_width//4*3, 0), (cam_width//4*3, cam_height), (255, 0, 0), 5)

    for item in items:
        if (item["x1"] + item["x2"]) / 2 < cam_width//4:
            item["pos"] = "left"
            # cv2.rectangle(image, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (255, 255, 255), 10)
        elif (item["x1"] + item["x2"]) / 2 > cam_width//4*3:
            item["pos"] = "right"
            # cv2.rectangle(image, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (255, 255, 255), 10)
        else:
            item["pos"] = "center"
            # cv2.rectangle(image, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (0, 0, 255), 10)

def clean_up(cap):
    cv2.destroyAllWindows()
    cap.release()