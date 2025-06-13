import requests
import cv2
import numpy as np
from ultralytics import YOLO


def coin_measurement(image, coin_results):
    coin_diameter_cm = 2.7

    if not coin_results or len(coin_results[0].boxes.xywh) == 0:
        return None, image

    boxes = coin_results[0].boxes.xywh
    areas = [w * h for x, y, w, h in boxes]
    smallest_idx = np.argmin(areas)
    x_coin, y_coin, w_coin, h_coin = boxes[smallest_idx]

    x1 = int(x_coin - w_coin / 2)
    y1 = int(y_coin - h_coin / 2)
    x2 = int(x_coin + w_coin / 2)
    y2 = int(y_coin + h_coin / 2)
    crop_coin = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop_coin, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    detected_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=100,
    )

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        a, b, r = detected_circles[0, 0]
        coin_diameter_px = 2 * r
        cv2.circle(image, (x1 + a, y1 + b), r, (255, 0, 0), 2)
    else:
        coin_diameter_px = min(w_coin, h_coin)

    scale_factor = coin_diameter_cm / coin_diameter_px
    return scale_factor, image


def baby_measurement(image, scale_factor, pose_model):
    pose_results = pose_model(image, stream=False)
    image1 = pose_results[0].plot()
    keypoints = pose_results[0].keypoints.xy[0]

    indices = np.array([0, 5, 15])
    selected_keypoints = keypoints[indices]
    nose, left_shoulder, left_ankle = selected_keypoints

    if any(np.isnan(k).any() for k in [nose, left_shoulder, left_ankle]):
        return None, image1

    length_px = np.linalg.norm(left_ankle - nose)
    baby_length_cm = length_px * scale_factor
    return baby_length_cm, image1


def measure_all(input_path, output_path):
    pose_model = YOLO("keypoints/yolo11s-pose.pt")
    coin_model = YOLO("coin/yolo11s.pt")
    image = cv2.imread(input_path)

    if image is None or len(image.shape) != 3 or image.shape[-1] != 3:
        return None, None

    coin_results = coin_model(image, stream=False)
    if not coin_results:
        return None, None

    scale_factor, image_with_coin = coin_measurement(image, coin_results)
    if scale_factor is None:
        return None, None

    baby_length, final_image = baby_measurement(image_with_coin, scale_factor, pose_model)
    if baby_length is None:
        return None, None

    cv2.imwrite(output_path, final_image)

    return baby_length, output_path
