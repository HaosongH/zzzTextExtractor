import cv2
import numpy as np
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from ppocr.utils.logging import get_logger
import logging
import re
from PIL import Image, ImageDraw, ImageFont
logger = get_logger()
logger.setLevel(logging.ERROR)

def mask_white_background(frame, threshold=200):
    """
    只保留白色背景的区域
    :param frame: 输入的图像帧（BGR格式）
    :param threshold: 白色阈值，越高越严格（用于判断白色区域）
    :return: 处理后的图像，只包含白色背景上的内容
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 生成二值化掩码，白色区域为255，其他为0
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 应用掩码到原始图像，保留白色区域
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_frame

def resize_with_boxes(frame, boxes, width=800):
    """等比例缩小图像并保留框架"""
        # 绘制 OCR 识别框到缩小后的图像上
    for box in boxes:
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    # 获取原图大小
    height, original_width = frame.shape[:2]
    ratio = width / original_width  # 计算缩放比例
    new_height = int(height * ratio)
    
    # 等比例缩小图像
    resized_frame = cv2.resize(frame, (width, new_height))


    
    return resized_frame

def put_text_on_right(frame, texts, width = 800):
    """在右侧显示识别到的文本"""
    # 右侧空白区域的大小
    height, original_width = frame.shape[:2]
    ratio = width / original_width  # 计算缩放比例
    new_height = int(height * ratio)
    right_frame = np.zeros((new_height, width // 2, 3), dtype=np.uint8)

    # 使用PIL绘制文本
    pil_image = Image.fromarray(cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 设置字体
    font_path = "pangmenzhengdao.ttf"  # 替换为你的字体文件路径
    font = ImageFont.truetype(font_path, 12)  # 设置字体和大小
    
    # 设置文本位置
    y_offset = 10
    for text in texts:
        draw.text((10, y_offset), text, font=font, fill=(255, 0, 0))  # 红色文字
        y_offset += 40  # 调整文本间距

    # 转换回OpenCV格式
    right_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return right_frame

class VideoOCRProcessor:
    def __init__(self, video_path, output_txt, output_screenshot, show_video=True, show_delay=30, frame_step = 30, screenshot_save = True):
        self.video_path = video_path
        self.output_txt = output_txt
        self.show_video = show_video
        self.show_delay = show_delay
        self.frame_step = frame_step
        self.output_screenshot = output_screenshot
        self.screenshot_save = screenshot_save
        # 初始化 PaddleOCR
        self.ocr = PaddleOCR(lang='ch',drop_score=0.8)

        # 存储识别的文本数据
        self.dialogues = []
        self.text_cache = []
        self.box_cache = []
        self.frame_num_cache = 0
        # 打开视频
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception(f"无法打开视频：{self.video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def on_trackbar(self, pos):
        """滑动条回调函数，跳转到指定帧"""
        self.current_frame = pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    
    def filterBoxText(self, boxes, texts):
        # Define blacklist conditions
        blackListStartwith = ["泛音","天入","天人",":","使敌人失衡"]
        blackListFull = ["三", ">", ">>","自动","目动","菜单","》","回顾","已收录","教学","G查看","直播"]
        blackListLen2 = ["自","动","跳","过","菜","单","启","司","式","回","顾","已","收","录","教","学","查","看"]
        # Create a set of indices that need to be removed based on blacklist conditions
        indices_to_remove = {
            i for i, text in enumerate(texts)
            if any(text.startswith(prefix) for prefix in blackListStartwith) or
            any(text == black_term for black_term in blackListFull) or 
            re.fullmatch(r"[A-Za-z0-9/\-\\+!@#$%^&*()_=\[\]{}|:;\"'<>,.? 。！·：一]+", text) or 
            len(text) == 1 or 
            (len(text)==2 and any(text.startswith(black_term) for black_term in blackListLen2)) or 
            (len(text)==2 and any(text.endswith(black_term) for black_term in blackListLen2)) 
        }
        
        # Use list comprehensions to filter out the indices to remove
        filtered_texts = [text for i, text in enumerate(texts) if i not in indices_to_remove]
        filtered_boxes = [box for i, box in enumerate(boxes) if i not in indices_to_remove]
        
        return filtered_boxes, filtered_texts
    
    def check_scene_update(self, input_text):
        if len(input_text) < len(self.text_cache):
            return True
        return False
    
    def scene_update(self):
        return 
    
    def line_update(self, input_box, input_text):
        self.text_cache = input_text
        self.box_cache = input_box
    
    def merge_nearby_texts(self, boxes, texts, y_threshold=300, x_threshold=300):
        """合并相近文本框的内容"""
        if not texts:
            return []
        character_list = ["安比","比利"]
        # 计算文本框的中心点和高度
        box_data = []
        for i, box in enumerate(boxes):
            x1, y1 = box[0]  # 左上角
            x2, y2 = box[2]  # 右下角
            center_y = (y1 + y2) / 2  # 计算中心点
            center_x = (x1 + x2) / 2
            height = y2 - y1
            box_data.append((x1, y1, x2, y2, center_y, height, texts[i], center_x))

        # 按 y 坐标排序
        box_data.sort(key=lambda x: (x[4], x[7]))  # 根据中心 y 坐标和 x 坐标排序

        merged_texts = []
        visited = [False] * len(box_data)

        for i in range(len(box_data)):
            if visited[i]:
                continue

            current_line = [box_data[i][6]]  # 当前行的第一个文本
            visited[i] = True
            # 如果文本为单独的角色名字，则是说话人名称，不与临近文本合并
            if current_line in character_list:
                continue
            for j in range(i + 1, len(box_data)):
                if visited[j]:
                    continue

                x1, y1, x2, y2, center_y, height, text, center_x = box_data[j]
                prev_x1, prev_y1, prev_x2, prev_y2, prev_center_y, prev_height, prev_text, prev_center_x = box_data[i]
                # print(f"prev_center_y: {prev_center_y}, center_y: {center_y}, prev_center_x: {prev_center_x}, center_x: {center_x}, text prev: {prev_text}, text: {text}")
                # 判断是否合并，x 和 y 方向都要满足阈值
                if abs(center_y - prev_center_y) < y_threshold and abs(center_x - prev_center_x) < x_threshold:
                    # print(f"合并文本：{prev_text} + {text}")
                    current_line.append(text)
                    visited[j] = True

            # 合并当前行的文本
            merged_texts.append(" ".join(current_line))
        return merged_texts

        
    def extract_text(self, frame):
        """使用 PaddleOCR 识别文本"""
        # 转换格式（OpenCV → PIL）
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # OCR 识别
        result = self.ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return [], []
        # 提取文本框
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
        probs = [line[1][1] for line in result[0]]
        return boxes, texts

    def frame_box_text_visualization(self, frame, boxes, texts):
        # 左侧是缩小后的原始图像和识别框
        left_frame = resize_with_boxes(frame, boxes)
        # 右侧显示识别到的文本
        right_frame = put_text_on_right(frame, texts)
        # 将左侧和右侧拼接在一起
        combined_frame = np.hstack((left_frame, right_frame))
        return combined_frame
        
        
    def process_video(self):
        """处理视频帧并进行 OCR 识别"""
        if self.show_video:
            cv2.namedWindow("Video Processing")
            cv2.createTrackbar("Frame", "Video Processing", 0, self.total_frames - 1, self.on_trackbar)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            frame = mask_white_background(frame, threshold=220)
            # OCR 识别
            boxes, texts = self.extract_text(frame)
            # 过滤无效文本
            boxes, texts = self.filterBoxText(boxes, texts)
            # 检测场景切换，并输出上一场景的文本
            if self.check_scene_update(texts):
                self.write_output()
            self.line_update(boxes, texts)
            # 实时显示
            if self.show_video:
                combined_frame = self.frame_box_text_visualization(frame, boxes, texts)
                
                # 显示拼接后的结果
                cv2.putText(combined_frame, f"Frame: {self.current_frame}/{self.total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow("Video Processing", combined_frame)
                key = cv2.waitKey(self.show_delay) & 0xFF
                if key == 27:
                    break
            self.frame_num_cache = self.current_frame
            self.on_trackbar(self.current_frame + self.frame_step - 1)
        self.write_output()    
        self.cap.release()
        cv2.destroyAllWindows()
        

    def write_output(self):
        """将识别结果保存到文件"""
        cleaned_text = self.merge_nearby_texts(self.box_cache, self.text_cache)
        cleaned_text = [s.replace(" ", "") for s in cleaned_text]
        self.text_cache = cleaned_text
        with open(self.output_txt, 'a+', encoding='utf-8') as f:
            f.write(f"{self.frame_num_cache} {self.text_cache}\n")
        if self.screenshot_save:
            self.on_trackbar(self.frame_num_cache)
            ret, frame = self.cap.read()
            left_frame = resize_with_boxes(frame, self.box_cache)
            success, encoded_image = cv2.imencode('.jpg', left_frame)
            if success:
                encoded_image.tofile(os.path.join(self.output_screenshot, f"{self.current_frame}.png"))
            self.on_trackbar(self.current_frame)
        print(f"{self.text_cache}")
        self.texts_cache = []
        self.box_cache = []
        

if __name__ == "__main__":
    video_folder = "../video/Story"  # 替换为你的视频文件夹路径
    output_folder = "../output"
    output_txt_folder = os.path.join(output_folder,"raw_txt")
    output_screenshot_folder = os.path.join(output_folder,"screenshot")
    os.makedirs(output_txt_folder, exist_ok=True)
    os.makedirs(output_screenshot_folder, exist_ok=True)
    for video_name in os.listdir(video_folder):
        if video_name.endswith('.mp4'):
            chapter_name, _ = os.path.splitext(os.path.basename(video_name))
            print(chapter_name)
            output_txt_path = os.path.join(output_txt_folder, f"{chapter_name}.txt")
            output_screenshot_chapter_folder = os.path.join(output_screenshot_folder, chapter_name)
            with open(output_txt_path, 'w+', encoding='utf-8') as f:
                pass
            os.makedirs(output_screenshot_chapter_folder, exist_ok=True)
            video_path = os.path.join(video_folder, video_name)
            
            processor = VideoOCRProcessor(video_path, output_txt=output_txt_path, output_screenshot = output_screenshot_chapter_folder, show_video=1, show_delay=30, screenshot_save = True)
            processor.process_video()

