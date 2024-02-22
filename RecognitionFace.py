import cv2
import numpy as np
import pyodbc
from keras.models import load_model

# Load mô hình CNN đã đào tạo để nhận diện khuôn mặt
model = load_model('C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/model.keras')

# Khởi tạo webcam
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

# Khởi tạo một đối tượng CascadeClassifier để phát hiện khuôn mặt trong hình ảnh đầu vào
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kết nối đến cơ sở dữ liệu SQL Server
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-N0U3R0S8;DATABASE=QLHSBN;Trusted_Connection=yes;')
cursor = conn.cursor()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong hình ảnh
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (128, 128))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)  # Chuyển ảnh về định dạng 3 kênh (RGB)
        face_roi = np.expand_dims(face_roi, axis=0)  # Thêm chiều cho batch

        # Dự đoán sử dụng mô hình CNN
        prediction = model.predict(face_roi)
        predicted_class = np.argmax(prediction)

        # Tìm kiếm trong cơ sở dữ liệu với ID được dự đoán
        cursor.execute("SELECT * FROM users WHERE id = ?", (int(predicted_class),))  # Chuyển đổi sang kiểu int
        patient_info = cursor.fetchone()

        if patient_info:
            print(f"Found patient with ID {predicted_class}. Patient Information: {patient_info}")
            # Hiển thị thông tin bệnh nhân trên hình ảnh (ví dụ: tên bệnh nhân)
            cv2.putText(img, f"Patient: {patient_info[1]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print(f"No patient found with ID {predicted_class}.")

        # Vẽ hình chữ nhật quanh khuôn mặt được phát hiện
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ khi kết thúc
cam.release()
cv2.destroyAllWindows()
