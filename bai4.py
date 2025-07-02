import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu, threshold_local
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing, rotate, zoom # zoom để scale

# --- Helper functions cho menu (tất cả đều KHÔNG dùng cv2) ---

def load_image_as_rgb_np(image_path):
    """Tải ảnh và chuyển đổi sang NumPy array RGB."""
    try:
        img_pil = Image.open(image_path)
        img_rgb_np = np.array(img_pil.convert('RGB'))
        return img_rgb_np
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy ảnh tại {image_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi tải ảnh: {e}")
        return None

def convert_rgb_to_gray(rgb_np_array):
    """Chuyển đổi ảnh RGB (NumPy array) sang Grayscale (NumPy array)."""
    # Công thức chuyển RGB sang Grayscale: 0.2989*R + 0.5870*G + 0.1140*B
    return np.dot(rgb_np_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

def perform_rotate(image_np, angle):
    """Xoay ảnh."""
    # scipy.ndimage.rotate tự động điều chỉnh kích thước để không bị cắt xén
    # order=1 là nội suy song tuyến (bilinear)
    rotated_img = rotate(image_np, angle=angle, reshape=True, order=1, mode='constant', cval=0)
    # rotate có thể trả về float, cần chuyển lại uint8
    if rotated_img.dtype != np.uint8:
        rotated_img = (rotated_img / rotated_img.max() * 255).astype(np.uint8) # Chuẩn hóa nếu cần
    return rotated_img

def perform_shift(image_np, dx, dy):
    """Tịnh tiến ảnh."""
    shifted_img = np.zeros_like(image_np)
    h, w = image_np.shape[:2]

    # Tính toán vùng nguồn và đích
    src_x1, src_y1 = max(0, -dx), max(0, -dy)
    src_x2, src_y2 = min(w, w - dx), min(h, h - dy)

    dest_x1, dest_y1 = max(0, dx), max(0, dy)
    dest_x2, dest_y2 = min(w, w + dx), min(h, h + dy)

    # Đảm bảo kích thước khớp nhau
    w_copy = min(src_x2 - src_x1, dest_x2 - dest_x1)
    h_copy = min(src_y2 - src_y1, dest_y2 - dest_y1)

    if w_copy > 0 and h_copy > 0:
        shifted_img[dest_y1 : dest_y1 + h_copy, dest_x1 : dest_x1 + w_copy] = \
            image_np[src_y1 : src_y1 + h_copy, src_x1 : src_x1 + w_copy]
    
    return shifted_img

def perform_scale(image_np, scale_factor):
    """Thay đổi kích thước ảnh."""
    # Sử dụng scipy.ndimage.zoom cho scaling
    # Nếu ảnh là màu (3 kênh), cần zoom từng kênh
    if len(image_np.shape) == 3:
        scaled_img = np.zeros((int(image_np.shape[0]*scale_factor), int(image_np.shape[1]*scale_factor), image_np.shape[2]), dtype=image_np.dtype)
        for i in range(image_np.shape[2]):
            scaled_img[:,:,i] = zoom(image_np[:,:,i], zoom=scale_factor, order=1) # order=1 là nội suy tuyến tính
    else: # Ảnh grayscale
        scaled_img = zoom(image_np, zoom=scale_factor, order=1)
    
    scaled_img = scaled_img.astype(np.uint8) # Đảm bảo kiểu dữ liệu uint8
    return scaled_img

def perform_adaptive_thresholding(image_np_gray, block_size, offset):
    """Áp dụng Adaptive Thresholding."""
    if block_size % 2 == 0: # block_size phải là số lẻ
        block_size += 1
        print(f"Điều chỉnh block_size thành {block_size} (số lẻ gần nhất).")
    
    adaptive_thresh_values = threshold_local(image_np_gray, block_size=block_size, offset=offset)
    binary_adaptive = (image_np_gray > adaptive_thresh_values).astype(np.uint8) * 255
    return binary_adaptive

def perform_binary_dilation(image_np_binary, iterations):
    """Thực hiện Binary Dilation."""
    # Đảm bảo ảnh đầu vào là boolean cho scipy.ndimage
    img_bool = image_np_binary.astype(bool)
    dilated_img = binary_dilation(img_bool, iterations=iterations).astype(np.uint8) * 255
