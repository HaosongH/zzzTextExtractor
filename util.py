import cv2
import numpy as np
def save_image(dest, image):
    # 将 RGB 图像转换为 BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 保存图像
    cv2.imwrite(dest, img_bgr)
    
def preprocess_image_binary(image):
    """对输入图像进行预处理以提高 OCR 识别效果"""
    # 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值去除背景
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return binary

def display_image(image):
        # Display the image in a window
    cv2.imshow('Displayed Image', image)

    # Wait for a key press and close the window
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Close all OpenCV windows
    
def mask_white_only(image):
    thresh = 220
    # 定义白色区域的阈值（R, G, B 都大于 200 认为是白色）
    lower_bound = np.array([thresh, thresh, thresh], dtype=np.uint8)
    upper_bound = np.array([255, 255, 255], dtype=np.uint8)

    # 生成掩码（白色区域为 255，其他区域为 0）
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 只保留白色区域，其他部分变黑
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def laplacian_sharpening(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Laplacian 进行锐化
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(gray - 0.5 * laplacian)  # 调整锐化强度
    return sharp

def erosion(image):
    # 创建腐蚀核（可以调整大小和形状）
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 3x3 矩形核
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆核
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字核

    # 进行腐蚀操作
    eroded = cv2.erode(image, kernel, iterations=1)  # iterations 控制腐蚀强度
    return eroded