import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import re
import ast
import locale
locale.setlocale(locale.LC_COLLATE, 'zh_CN.UTF-8')  # 设置中文排序规则
class FrameEditor:
    def __init__(self, master, txt_files, png_folder_root, output_folder):
        """
        :param txt_files: 所有待处理的 txt 文件路径列表
        :param png_folder_root: png 文件根目录，每个 txt 文件对应的 png 文件夹名称与 txt 文件名（去扩展名）相同
        :param output_folder: 保存校正后 txt 的文件夹
        """
        self.master = master
        self.txt_files = txt_files  # 列表
        self.png_folder_root = png_folder_root
        self.output_folder = output_folder

        self.current_file_index = 0   # 当前 txt 文件在列表中的索引
        self.txt_file = self.txt_files[self.current_file_index]
        chapter_name, _ = os.path.splitext(os.path.basename(self.txt_file))
        self.png_folder = os.path.join(self.png_folder_root, chapter_name)

        self.frames = []  # 存放所有帧号（字符串格式）
        self.data = {}    # {frame_number: [list of texts]}
        self.current_index = 0

        # 存放8个对话人输入框的引用，键为 '1'~'8'
        self.dialogue_entries = {}

        self.load_data()
        self.create_widgets()
        self.bind_keys()
        self.show_frame()

    def load_data(self):
        """加载当前 txt 文件数据，并解析每一行"""
        self.frames = []
        self.data = {}
        try:
            with open(self.txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败：{self.txt_file}\n{e}")
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 每行格式：数字 后跟空格，再跟 list 字符串
            m = re.match(r'^(\d+)\s+(\[.*\])$', line)
            if m:
                frame_number = m.group(1)
                try:
                    text_list = ast.literal_eval(m.group(2))
                except Exception as e:
                    text_list = []
                self.data[frame_number] = text_list
                self.frames.append(frame_number)
        # 根据帧号进行排序
        self.frames.sort(key=lambda x: int(x))
        self.current_index = 0

    def create_widgets(self):
        """创建 UI 控件"""
        self.master.title("视频帧校正工具")
        
        # 左侧用于显示图片
        self.image_label = tk.Label(self.master)
        self.image_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 右侧区域
        right_frame = tk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 对话人输入区域（8个文本框）
        dialogue_frame = tk.Frame(right_frame)
        dialogue_frame.pack(fill=tk.X, pady=5)
        tk.Label(dialogue_frame, text="对话人（按数字1～8添加）：").pack(side=tk.LEFT)
        for i in range(1, 9):
            entry = tk.Entry(dialogue_frame, width=10, font=("Arial", 12))
            entry.pack(side=tk.LEFT, padx=2)
            entry.insert(0, f"A{i}")
            self.dialogue_entries[str(i)] = entry
        
        # 主文本编辑区域（启用撤销功能）
        self.text_widget = tk.Text(right_frame, wrap=tk.WORD, font=("Arial", 14), undo=True)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        # 绑定 Ctrl+Z 撤销操作
        self.text_widget.bind("<Control-z>", lambda event: self.text_widget.edit_undo())
        
        # 按钮区域
        button_frame = tk.Frame(right_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.prev_button = tk.Button(button_frame, text="上一个", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = tk.Button(button_frame, text="下一个", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_button = tk.Button(button_frame, text="删除本帧", command=self.delete_current_frame, fg="red")
        self.delete_button.pack(side=tk.LEFT, padx=5)
        
        # 新增下一个文件按钮
        self.next_file_button = tk.Button(button_frame, text="下一个文件", command=self.load_next_file, fg="blue")
        self.next_file_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(button_frame, text="保存修改", command=self.save_changes)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
    def bind_keys(self):
        """绑定Control+数字键事件，用于在光标处插入对应对话人文本"""
        # 显示快捷键提示信息
        shortcut_info = tk.Label(self.master, text="使用 Control+数字键（1-8）插入对应对话人文本")
        shortcut_info.pack(pady=5)
        # 绑定 Control+数字键
        for i in range(1, 9):
            self.master.bind(f"<Control-Key-{i}>", self.handle_key_press)
    
    def handle_key_press(self, event):
        """处理Control+数字键事件，在光标处插入对应对话人文本"""
        key = event.keysym
        if key.isdigit() and 1 <= int(key) <= 8:
            dialogue = self.dialogue_entries.get(key).get().strip()
            if dialogue:
                # 获取当前光标位置
                cursor_position = self.text_widget.index(tk.INSERT)
                self.text_widget.insert(cursor_position, dialogue + "\n")
                # 使焦点保持在文本框中
                self.text_widget.focus_set()


    def show_frame(self):
        """显示当前帧的图片和文本"""
        if not self.frames:
            self.image_label.config(image="", text="无图片")
            self.text_widget.delete(1.0, tk.END)
            self.master.title("视频帧校正工具 - 无帧")
            return

        frame_number = self.frames[self.current_index]
        image_path = os.path.join(self.png_folder, f"{frame_number}.png")
        if os.path.exists(image_path):
            pil_image = Image.open(image_path)
            # 等比例缩放图片到指定大小（此处设定宽度为800，高度自适应）
            original_width, original_height = pil_image.size
            new_width = 800
            new_height = int(original_height * new_width / original_width)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=self.photo)
        else:
            self.image_label.config(image="", text="无图片")
        
        # 更新文本编辑区域
        self.text_widget.delete(1.0, tk.END)
        text_list = self.data.get(frame_number, [])
        for s in text_list:
            self.text_widget.insert(tk.END, s + "\n")
        
        self.master.title(f"视频帧校正工具 - 当前帧：{frame_number}   文件：{os.path.basename(self.txt_file)}")

    def save_current_frame_text(self):
        """保存当前帧编辑的文本内容到数据字典中"""
        if not self.frames:
            return
        frame_number = self.frames[self.current_index]
        content = self.text_widget.get(1.0, tk.END).strip()
        lines = [line for line in content.splitlines() if line.strip()]
        self.data[frame_number] = lines

    def prev_frame(self):
        """切换到上一帧"""
        self.save_current_frame_text()
        if self.current_index > 0:
            self.current_index -= 1
            self.show_frame()
        else:
            messagebox.showinfo("提示", "已经是第一帧")

    def next_frame(self):
        """切换到下一帧"""
        self.save_current_frame_text()
        if self.current_index < len(self.frames) - 1:
            self.current_index += 1
            self.show_frame()
        else:
            messagebox.showinfo("提示", "已经是最后一帧")

    def delete_current_frame(self):
        """删除当前帧及其对应的文本记录"""
        if not self.frames:
            return
        frame_number = self.frames[self.current_index]
        if not messagebox.askyesno("确认删除", f"确定要删除帧 {frame_number} 及其对应的文本记录吗？"):
            return
        del self.data[frame_number]
        self.frames.pop(self.current_index)
        messagebox.showinfo("提示", f"帧 {frame_number} 已删除。")
        if self.current_index >= len(self.frames) and self.current_index > 0:
            self.current_index -= 1
        self.show_frame()

    def load_next_file(self):
        """保存当前文件并加载下一个文件"""
        self.save_current_frame_text()
        # 保存当前文件数据到原 txt 文件所在路径
        with open(self.txt_file, "w", encoding="utf-8") as f:
            for frame in self.frames:
                f.write(f"{frame} {self.data[frame]}\n")
        # 如果还有下一个文件，则加载
        if self.current_file_index < len(self.txt_files) - 1:
            self.current_file_index += 1
            self.txt_file = self.txt_files[self.current_file_index]
            chapter_name, _ = os.path.splitext(os.path.basename(self.txt_file))
            self.png_folder = os.path.join(self.png_folder_root, chapter_name)
            self.load_data()
            self.show_frame()
        else:
            messagebox.showinfo("提示", "已经是最后一个文件。")

    def save_changes(self):
        """将所有校正后的内容保存到目标文件夹中，文件名与输入文件相同"""
        self.save_current_frame_text()
        save_path = os.path.join(self.output_folder, os.path.basename(self.txt_file))
        with open(save_path, "w", encoding="utf-8") as f:
            for frame in self.frames:
                frame_text_list = self.data[frame]
                for text in frame_text_list:
                    # 可根据需求自定义写入格式
                    f.write(f"{text}\n")
        messagebox.showinfo("提示", f"保存成功：{save_path}")

if __name__ == "__main__":
    # 定义相关文件夹路径
    output_folder = "../output"
    txt_folder = os.path.join(output_folder, "txt")
    screenshot_folder = os.path.join(output_folder, "screenshot")
    cleaned_output_folder = "../completed"
    os.makedirs(cleaned_output_folder, exist_ok=True)

    # 获取所有 txt 文件
    txt_files = []
    for file in os.listdir(txt_folder):
        if file.endswith(".txt"):
            txt_files.append(os.path.join(txt_folder, file))
    txt_files.sort(key=lambda x: locale.strxfrm(os.path.splitext(os.path.basename(x))[0]))

    root = tk.Tk()
    app = FrameEditor(root, txt_files, screenshot_folder, cleaned_output_folder)
    root.mainloop()
