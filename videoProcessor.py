import cv2
import numpy as np
import os
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.ERROR)

blackList = ["泛式","三"]

class VideoOCRProcessor:
    def __init__(self, video_path, output_txt='output.txt', show_video=True, show_delay=30):
        self.video_path = video_path
        self.output_txt = output_txt
        self.show_video = show_video
        self.show_delay = show_delay
        
        # 初始化 PaddleOCR
        self.ocr = PaddleOCR(lang='ch')

        # 存储识别的文本数据
        self.dialogues = []
        self.last_speaker = "Unknown"
        self.last_text = ""

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

    def extract_text(self, frame):
        """使用 PaddleOCR 识别文本"""
        # 转换格式（OpenCV → PIL）
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # OCR 识别
        result = self.ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return [], []
        print(result)
        exit()
        # 提取文本框
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
        return boxes, texts

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

            # OCR 识别
            boxes, texts = self.extract_text(frame)

            # 识别说话人及台词
            if texts:
                speaker = texts[0] if len(texts) > 1 else "Unknown"
                dialogue = " ".join(texts[1:]) if len(texts) > 1 else texts[0]

                if dialogue != self.last_text:
                    self.dialogues.append((speaker, dialogue))
                    self.last_text = dialogue
                    self.last_speaker = speaker

            # 实时显示
            if self.show_video:
                # 绘制 OCR 识别框
                for box, text in zip(boxes, texts):
                    pts = np.array(box, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(frame, text, (pts[0][0][0], pts[0][0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 显示当前帧
                cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow("Video Processing", frame)
                key = cv2.waitKey(self.show_delay) & 0xFF
                if key == 27:
                    break

        self.cap.release()
        cv2.destroyAllWindows()
        self.write_output()

    def write_output(self):
        """将识别结果保存到文件"""
        with open(self.output_txt, 'w', encoding='utf-8') as f:
            for speaker, dialogue in self.dialogues:
                f.write(f"{speaker}: {dialogue}\n")
        print(f"✅ OCR 结果已保存至 {self.output_txt}")

if __name__ == "__main__":
    video_folder = "../video/Story"  # 替换为你的视频文件夹路径
    for video_name in os.listdir(video_folder):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_name)
            processor = VideoOCRProcessor(video_path, output_txt='output.txt', show_video=1, show_delay=30)
            processor.process_video()

