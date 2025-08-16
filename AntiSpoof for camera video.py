import cv2
import numpy as np
import onnxruntime as ort

# ==============================
# Load YOLOv5 face detector
# ==============================
yolo_model_path = "yolov5s-face.onnx"
yolo_session = ort.InferenceSession(yolo_model_path, providers=["CPUExecutionProvider"])
yolo_input_name = yolo_session.get_inputs()[0].name
yolo_output_name = yolo_session.get_outputs()[0].name

# ==============================
# Load Anti-Spoofing model
# ==============================
antispoof_model_path = "AntiSpoofing_bin_1.5_128 (2).onnx"
spoof_session = ort.InferenceSession(antispoof_model_path, providers=["CPUExecutionProvider"])
spoof_input_name = spoof_session.get_inputs()[0].name
spoof_output_name = spoof_session.get_outputs()[0].name

# ==============================
# Helper: YOLOv5 NMS
# ==============================
def non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.5):
    boxes, confidences, class_ids = [], [], []
    for det in prediction:
        if det[4] > conf_thres:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * det[4]
            if confidence > conf_thres:
                boxes.append(det[:4])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    if not boxes:
        return []

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    # Convert xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    idxs = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), conf_thres, iou_thres)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            results.append((boxes_xyxy[i], confidences[i], class_ids[i]))
    return results

# ==============================
# Face detection
# ==============================
def detect_face(image, conf_threshold=0.2):
    img_h, img_w = image.shape[:2]
    img_resized = cv2.resize(image, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    img_rgb = np.expand_dims(img_rgb, 0)

    pred = yolo_session.run([yolo_output_name], {yolo_input_name: img_rgb})[0][0]
    detections = non_max_suppression(pred, conf_thres=conf_threshold)

    boxes_out = []
    for box_xyxy, conf, cls in detections:
        x1 = int(box_xyxy[0] / 640 * img_w)
        y1 = int(box_xyxy[1] / 640 * img_h)
        x2 = int(box_xyxy[2] / 640 * img_w)
        y2 = int(box_xyxy[3] / 640 * img_h)
        boxes_out.append((x1, y1, x2, y2, conf, cls))
    return boxes_out

# ==============================
# Anti-spoofing
# ==============================
def detect_spoof(face_img):
    face_resized = cv2.resize(face_img, (128, 128))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_rgb = face_rgb.astype(np.float32) / 255.0
    face_rgb = np.transpose(face_rgb, (2, 0, 1))
    face_rgb = np.expand_dims(face_rgb, 0)

    preds = spoof_session.run([spoof_output_name], {spoof_input_name: face_rgb})[0]
    label = int(np.argmax(preds))
    return "Spoof" if label == 1 else "Real"

# ==============================
# Run on Camera
# ==============================
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_face(frame)

    for (x1, y1, x2, y2, conf, cls) in detections:
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        result = detect_spoof(face_img)

        color = (0, 255, 0) if result == "Real" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{result} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Anti-Spoofing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
