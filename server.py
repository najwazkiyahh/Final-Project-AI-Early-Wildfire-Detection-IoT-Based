import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Load YOLOv5 model (pretrained)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. Load Teachable Machine model (.h5)
keras_model = load_model('C:/Users/LENOVO/Downloads/foreguard/hutan33/hutan3/converted_keras/keras_model.h5')
labels = ["Forest", "Forest_Fire"]
IMG_SIZE = 224

# 3. Preprocessing untuk Teachable Machine
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return np.expand_dims(img, axis=0)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    print("Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        #### ==== YOLOv5 Detection ==== ####
        yolo_results = yolo_model(frame)
        boxes = yolo_results.pandas().xyxy[0]

        for _, row in boxes.iterrows():
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']
            conf = row['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #### ==== Teachable Machine Detection ==== ####
        h, w, _ = frame.shape
        size = min(h, w)
        center_crop = frame[h//2 - size//2:h//2 + size//2, w//2 - size//2:w//2 + size//2]
        batch = preprocess(center_crop)
        pred = keras_model.predict(batch)[0]
        idx = np.argmax(pred)
        score = pred[idx]
        tm_label = labels[idx]

        # Tampilkan label Teachable Machine
        cv2.putText(frame, f"TM: {tm_label} ({score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if idx == 1 else (0, 255, 0), 2)

        #### ==== Show Frame ==== ####
        cv2.imshow("Deteksi Gabungan: YOLOv5 + TM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
