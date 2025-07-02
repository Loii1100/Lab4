import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image # Để đọc ảnh và chuyển đổi chế độ dễ dàng

def process_ho_xuan_huong_with_crop():
    # --- Cấu hình ---
    IMAGE_PATH = 'dalat.jpg' # Đảm bảo tên file này khớp với ảnh của bạn
    ROTATION_ANGLE = 45 # Góc xoay 45 độ
    ADAPTIVE_BLOCK_SIZE = 61 # Kích thước khối cho Adaptive Thresholding (phải là số lẻ)
    ADAPTIVE_OFFSET = 10 # Offset (C) cho Adaptive Thresholding

    # --- 1. Đọc ảnh và cắt vùng Hồ Xuân Hương ---
    try:
        full_img_bgr = cv2.imread(IMAGE_PATH)
        if full_img_bgr is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {IMAGE_PATH}. Đảm bảo ảnh nằm cùng thư mục.")
        
        full_img_rgb = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2RGB)
        
        img_height, img_width, _ = full_img_bgr.shape

        # Ước lượng vùng cắt cho Hồ Xuân Hương từ ảnh 'dalat.jpg' bạn đã cung cấp:
        # Hồ Xuân Hương có vẻ nằm ở khoảng giữa bên trái của ảnh collage.
        # Bạn cần điều chỉnh các giá trị này dựa trên kích thước và vị trí thực tế của Hồ Xuân Hương
        # trong ảnh dalat.jpg của bạn.
        
        # Ví dụ ước lượng (điều chỉnh nếu cần):
        # x_crop, y_crop, w_crop, h_crop
        x_crop = int(img_width * 0.33) # Bắt đầu từ 1/3 chiều rộng
        y_crop = int(img_height * 0.25) # Bắt đầu từ 1/4 chiều cao
        w_crop = int(img_width * 0.33) # Rộng khoảng 1/3 ảnh
        h_crop = int(img_height * 0.5)  # Cao khoảng 1/2 ảnh
        
        ho_xuan_huong_region_bgr = full_img_bgr[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop]
        
        if ho_xuan_huong_region_bgr.size == 0:
            print("Lỗi: Vùng Hồ Xuân Hương được cắt ra rỗng. Vui lòng kiểm tra lại tọa độ cắt (x_crop, y_crop, w_crop, h_crop).")
            return
        
        # Chuyển vùng Hồ Xuân Hương đã cắt sang ảnh xám (PIL Image để dùng skimage.filters)
        # Hoặc dùng cv2.cvtColor(ho_xuan_huong_region_bgr, cv2.COLOR_BGR2GRAY) nếu dùng cv2.adaptiveThreshold
        # Vì đề bài gợi ý dùng threshold_local (skimage), chúng ta sẽ dùng PIL Image
        ho_xuan_huong_gray_pil = Image.fromarray(cv2.cvtColor(ho_xuan_huong_region_bgr, cv2.COLOR_BGR2RGB)).convert('L')
        ho_xuan_huong_gray_np = np.array(ho_xuan_huong_gray_pil)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Vui lòng kiểm tra lại đường dẫn và tên tệp '{IMAGE_PATH}'.")
        print("Đảm bảo ảnh nằm cùng thư mục với script này.")
        return
    except Exception as e:
        print(f"Lỗi khi đọc, cắt hoặc chuyển đổi ảnh: {e}")
        return

    # --- 2. Áp dụng Adaptive Thresholding cho vùng Hồ Xuân Hương đã cắt ---
    try:
        # threshold_local từ skimage.filters
        adaptive_thresh_img_values = threshold_local(ho_xuan_huong_gray_np, 
                                                     block_size=ADAPTIVE_BLOCK_SIZE, 
                                                     offset=ADAPTIVE_OFFSET)
        
        # Tạo ảnh nhị phân
        binary_adaptive_ho_xuan_huong = (ho_xuan_huong_gray_np > adaptive_thresh_img_values).astype(np.uint8) * 255
    except Exception as e:
        print(f"Lỗi khi áp dụng Adaptive Thresholding: {e}")
        return

    # --- 3. Xoay đối tượng (vùng đã phân ngưỡng) 45 độ ---
    (h_obj, w_obj) = binary_adaptive_ho_xuan_huong.shape
    
    # Xác định tâm xoay (tâm của vùng Hồ Xuân Hương đã cắt)
    center = (w_obj // 2, h_obj // 2)
    
    # Lấy ma trận xoay
    M_rotate = cv2.getRotationMatrix2D(center, ROTATION_ANGLE, 1.0) # Góc 45 độ, tỷ lệ 1.0

    # Tính toán kích thước mới của ảnh sau khi xoay để không bị mất thông tin
    cos = np.abs(M_rotate[0, 0])
    sin = np.abs(M_rotate[0, 1])
    new_w = int((h_obj * sin) + (w_obj * cos))
    new_h = int((h_obj * cos) + (w_obj * sin))

    # Điều chỉnh ma trận tịnh tiến để ảnh xoay nằm ở giữa khung mới
    M_rotate[0, 2] += (new_w / 2) - center[0]
    M_rotate[1, 2] += (new_h / 2) - center[1]

    # Thực hiện xoay ảnh
    rotated_ho_xuan_huong = cv2.warpAffine(binary_adaptive_ho_xuan_huong, M_rotate, 
                                            (new_w, new_h), 
                                            borderMode=cv2.BORDER_CONSTANT, 
                                            borderValue=0) # Nền đen

    # --- 4. Hiển thị kết quả
