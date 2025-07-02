import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
# from scipy.ndimage import shift # Có thể dùng shift nhưng đơn giản hơn với slicing

def process_langbiang_no_cv2():
    # --- Cấu hình ---
    IMAGE_PATH = 'dalat.jpg' # Đảm bảo tên file này khớp với ảnh của bạn
    SHIFT_AMOUNT_X = 100 # Tịnh tiến sang phải 100 pixel

    # --- 1. Mở ảnh và cắt vùng LangBiang ---
    try:
        # Đọc ảnh bằng PIL
        full_img_pil = Image.open(IMAGE_PATH)
        full_img_rgb_np = np.array(full_img_pil.convert('RGB')) # Để hiển thị ảnh màu gốc

        img_height, img_width, _ = full_img_rgb_np.shape

        # Vùng cắt thử nghiệm cho LangBiang từ ảnh 'dalat.jpg' (góc trên bên trái)
        x_crop, y_crop, w_crop, h_crop = 0, 0, int(img_width * 0.35), int(img_height * 0.55)
        
        # Cắt vùng LangBiang từ ảnh gốc (NumPy array)
        # Chú ý: .copy() để đảm bảo tạo ra một bản sao độc lập
        langbiang_region_rgb = full_img_rgb_np[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop].copy()
        
        if langbiang_region_rgb.size == 0:
            print("Lỗi: Vùng LangBiang được cắt ra rỗng. Vui lòng kiểm tra lại tọa độ cắt.")
            return

        # Chuyển vùng LangBiang đã cắt sang ảnh xám
        # Công thức chuyển RGB sang Grayscale: 0.2989*R + 0.5870*G + 0.1140*B
        langbiang_gray = np.dot(langbiang_region_rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Vui lòng kiểm tra lại đường dẫn và tên tệp '{IMAGE_PATH}'.")
        return
    except Exception as e:
        print(f"Lỗi khi đọc, cắt hoặc chuyển đổi ảnh: {e}")
        return

    # --- 2. Áp dụng phân ngưỡng Otsu cho vùng LangBiang đã cắt ---
    try:
        thresh_val_otsu = threshold_otsu(langbiang_gray)
        binary_otsu_langbiang = (langbiang_gray > thresh_val_otsu).astype(np.uint8) * 255
    except Exception as e:
        print(f"Lỗi khi áp dụng phân ngưỡng Otsu: {e}")
        return

    # --- 3. Tịnh tiến vùng LangBiang đã phân ngưỡng ---
    # Tạo một ảnh nền đen có cùng kích thước với vùng LangBiang đã cắt ban đầu
    translated_langbiang_binary = np.zeros_like(binary_otsu_langbiang, dtype=np.uint8)

    # Tính toán vị trí mới
    new_x_start = SHIFT_AMOUNT_X
    new_y_start = 0 # Không dịch chuyển dọc

    # Đảm bảo không vượt quá biên ảnh
    # Tính toán kích thước phần ảnh có thể dán vào (nếu bị cắt biên)
    w_src = binary_otsu_langbiang.shape[1]
    h_src
