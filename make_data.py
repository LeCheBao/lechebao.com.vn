import cv2
import os
import pyodbc

# Kết nối đến cơ sở dữ liệu SQL Server
#conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-N0U3R0S8;DATABASE=QLHSBN;Trusted_Connection=yes;')
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=LAPTOP-N0U3R0S8;DATABASE=QLHSBN;Trusted_Connection=yes;', timeout=30)

cursor = conn.cursor()

# Nhập id và name người mình muốn thêm mặt vào
id = input("Nhập Id: ")
name = input("Nhập tên: ")
age = input("Nhập tuổi: ")
bloodgroup = input("Nhập nhóm máu: ")
disease = input("Nhập căn bệnh: ")

# Kiểm tra xem id đã tồn tại trong bảng users chưa
cursor.execute("SELECT id FROM users WHERE id = ?", id)
user_exists = cursor.fetchone()

if user_exists:
    print(f"Người dùng với id {id} đã tồn tại trong cơ sở dữ liệu.")
else:
    # Nếu id không tồn tại, thêm thông tin người dùng vào cơ sở dữ liệu
    cursor.execute("INSERT INTO users (id, name, age, bloodgroup, disease) VALUES (?, ?, ?, ?, ?)", (id, name, age, bloodgroup, disease))
    conn.commit()

    # Khởi tạo webcam và đặt độ phân giải của nó thành 1280x720
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)

    # Khởi tạo một đối tượng CascadeClassifier trong thư viện OpenCV với tệp tin XML chứa thông tin về mô hình Cascade để phát hiện khuôn mặt trên hình ảnh đầu vào
    detector = cv2.CascadeClassifier("C:/Users/VICTUS/Documents/Zalo Received Files/Artifical intelligence/Artifical intelligence/haarcascade_frontalface_default.xml")
    # Biến này sẽ được sử dụng để theo dõi số lượng ảnh khuôn mặt được chụp cho người dùng này.
    sampleNum = 0

    while True:
        # Đọc dữ liệu video từ máy ảnh và lưu trữ các khung hình trong biến img.
        ret, img = cam.read()
        # Chuyển đổi hình ảnh màu sang độ xám để đơn giản hóa trong việc phát hiện khuôn mặt
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Phát hiện khuôn mặt trong hình ảnh thang độ xám
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Vẽ một hình chữ nhật xung quanh khuôn mặt được phát hiện
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Tạo ra một thư mục có tên dataSet nếu nó chưa tồn tại
            if not os.path.exists('data'):
                os.makedirs('data')

            sampleNum += 1
            # Lưu khuôn mặt được phát hiện vào cơ sở dữ liệu
            image_path = "data/User." + id + '.' + str(sampleNum) + ".jpg"
            cv2.imwrite(image_path, gray[y:y+h, x:x+w])

            # Thêm thông tin ảnh vào cơ sở dữ liệu
            cursor.execute("INSERT INTO images (user_id, image_path) VALUES (?, ?)", (id, image_path))
            conn.commit()

            cv2.imshow('frame', img)

        # Nhấn phím q để kết thúc chương trình
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Nếu số ảnh lưu được đủ 250 ảnh thì dừng chương trình
        elif sampleNum > 250:
            break

    # Giải phóng máy ảnh và phá hủy tất cả các cửa sổ do chương trình tạo ra
    cam.release()
    cv2.destroyAllWindows()
