import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from PIL import Image # Sử dụng PIL để đọc ảnh và convert sang grayscale dễ dàng hơn

def process_langbiang_with_crop():
    # --- Cấu hình ---
    IMAGE_PATH = 'dalat.jpg' # Đảm bảo tên file này khớp với ảnh của bạn
    SHIFT_AMOUNT_X = 100 # Tịnh tiến sang phải 100 pixel

    # --- 1. Đọc ảnh và cắt vùng LangBiang ---
    try:
        # Đọc ảnh màu bằng OpenCV để có thể thao tác màu nếu cần
        full_img_bgr = cv2.imread(IMAGE_PATH)
        if full_img_bgr is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {IMAGE_PATH}. Đảm bảo ảnh nằm cùng thư mục.")
        
        # Chuyển ảnh sang RGB để hiển thị với matplotlib
        full_img_rgb = cv2.cvtColor(full_img_bgr, cv2.COLOR_BGR2RGB)

        # Xác định vùng LangBiang trong ảnh dalat.jpg (tọa độ ước lượng thủ công)
        # Bạn có thể cần điều chỉnh các giá trị này tùy theo kích thước chính xác của ảnh dalat.jpg của bạn
        # Cấu trúc: [y_start:y_end, x_start:x_end]
        # Quan sát ảnh dalat.jpg, LangBiang nằm ở góc trên bên trái.
        # Ví dụ: Giả sử vùng LangBiang là 1/3 chiều rộng và 1/2 chiều cao của ảnh gốc
        # Bạn có thể dùng công cụ xem ảnh để xác định pixel chính xác hơn
        
        # Để chính xác hơn, tôi sẽ ước lượng từ ảnh bạn cung cấp
        # (x_start, y_start) = (0, 0)
        # width = khoảng 1/3 chiều rộng ảnh
        # height = khoảng 1/2 chiều cao ảnh
        
        img_height, img_width, _ = full_img_bgr.shape
        
        # Ước lượng vùng cắt cho LangBiang từ ảnh 'dalat.jpg' bạn đã cung cấp:
        # Vùng LangBiang có vẻ nằm trong khoảng (0, 0) đến (chiều rộng / 3, chiều cao / 2)
        # Hoặc một tọa độ cụ thể hơn nếu bạn đã xác định được.
        # Ở đây, tôi sẽ ước lượng một ROI (Region of Interest) để cắt LangBiang ra.
        # Giả sử LangBiang chiếm khoảng 1/3 chiều rộng và 1/2 chiều cao từ góc trên bên trái
        
        # Ví dụ: Nếu ảnh của bạn là 1200x800 (chiều rộng x chiều cao)
        # x_start = 0, x_end = 400 (1/3 của 1200)
        # y_start = 0, y_end = 400 (1/2 của 800)
        
        # Bạn cần điều chỉnh các giá trị này dựa trên kích thước thực của ảnh `dalat.jpg`
        # và vị trí của LangBiang trong ảnh đó.
        # Để an toàn, tôi sẽ lấy một vùng lớn hơn một chút để đảm bảo bao trọn LangBiang
        
        # Vùng cắt thử nghiệm (điều chỉnh nếu cần):
        # x, y, w, h
        x_crop, y_crop, w_crop, h_crop = 0, 0, int(img_width * 0.35), int(img_height * 0.55)
        
        # Cắt vùng LangBiang từ ảnh gốc
        langbiang_region_bgr = full_img_bgr[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop]
        
        if langbiang_region_bgr.size == 0:
            print("Lỗi: Vùng LangBiang được cắt ra rỗng. Vui lòng kiểm tra lại tọa độ cắt (x_crop, y_crop, w_crop, h_crop).")
            return
        
        # Chuyển vùng LangBiang đã cắt sang ảnh xám để phân ngưỡng
        langbiang_gray = cv2.cvtColor(langbiang_region_bgr, cv2.COLOR_BGR2GRAY)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}. Vui lòng kiểm tra lại đường dẫn và tên tệp '{IMAGE_PATH}'.")
        print("Đảm bảo ảnh nằm cùng thư mục với script này.")
        return
    except Exception as e:
        print(f"Lỗi khi đọc, cắt hoặc chuyển đổi ảnh: {e}")
        return

    # --- 2. Áp dụng phân ngưỡng Otsu cho vùng LangBiang đã cắt ---
    try:
        # threshold_otsu trả về giá trị ngưỡng tối ưu
        thresh_val_otsu = threshold_otsu(langbiang_gray)
        # Tạo ảnh nhị phân: pixel nào lớn hơn ngưỡng thì thành trắng (255), ngược lại thành đen (0)
        binary_otsu_langbiang = (langbiang_gray > thresh_val_otsu).astype(np.uint8) * 255
    except Exception as e:
        print(f"Lỗi khi áp dụng phân ngưỡng Otsu cho vùng LangBiang: {e}")
        return

    # --- 3. Tịnh tiến vùng LangBiang đã phân ngưỡng ---
    # Ảnh đã được phân ngưỡng là `binary_otsu_langbiang`. Chúng ta sẽ tịnh tiến ảnh này.
    
    # Ma trận M cho tịnh tiến (dx, dy)
    M = np.float32([[1, 0, SHIFT_AMOUNT_X], [0, 1, 0]]) # dx=100, dy=0

    # Áp dụng phép tịnh tiến lên ảnh nhị phân LangBiang
    # Kích thước ảnh đích là kích thước của vùng LangBiang đã cắt
    translated_langbiang_binary = cv2.warpAffine(binary_otsu_langbiang, M, 
                                                (langbiang_gray.shape[1], langbiang_gray.shape[0]),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0) # Nền đen

    # --- 4. Hiển thị kết quả ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(full_img_rgb) # Hiển thị ảnh collage đầy đủ
    plt.title('Ảnh gốc (dalat.jpg)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(langbiang_gray, cmap='gray') # Hiển thị vùng LangBiang đã cắt (ảnh xám)
    plt.title('Vùng LangBiang đã cắt')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(translated_langbiang_binary, cmap='gray')
    plt.title(f'LangBiang đã phân vùng & tịnh tiến {SHIFT_AMOUNT_X}px phải')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- 5. Lưu ảnh kết quả ---
    output_filename = 'lang_biang.jpg'
    cv2.imwrite(output_filename, translated_langbiang_binary)
    print(f"Ảnh LangBiang đã tịnh tiến được lưu thành: {output_filename}")

# --- Gọi hàm để chạy bài 1 ---
if __name__ == "__main__":
    process_langbiang_with_crop()
