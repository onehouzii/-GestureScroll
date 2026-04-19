import sys
import cv2
import mediapipe as mp
import pyautogui
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from collections import deque, Counter

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    gesture_signal = pyqtSignal(str)
    
    # 手指关节索引定义：[近节/掌指关节, 远节/指间关节, 指尖]
    FINGER_JOINTS = {
        'thumb': [2, 3, 4],    # 拇指：MCP, IP, TIP
        'index': [6, 7, 8],    # 食指：PIP, DIP, TIP
        'middle': [10, 11, 12],# 中指：PIP, DIP, TIP
        'ring': [14, 15, 16],  # 无名指：PIP, DIP, TIP
        'pinky': [18, 19, 20]  # 小指：PIP, DIP, TIP
    }
    
    def __init__(self):
        super().__init__()
        self.running = False
        
        # MediaPipe配置，提高置信度过滤低质量检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 防抖与状态跟踪
        self.gesture_history = deque(maxlen=10)  # 扩大历史窗口，增强防抖
        self.last_valid_gesture = ""  # 上一次有效手势，用于边缘触发
        
        # 空帧防抖：防止偶尔的检测波动导致状态重置
        self.empty_frame_count = 0
        self.EMPTY_THRESHOLD = 3  # 连续3帧空手势，才会真正重置状态
    
    def calculate_finger_angle(self, landmarks, joint_idxs):
        """
        计算手指三个关节形成的夹角，用于判断手指弯曲程度
        不受手部朝向影响，鲁棒性更强
        """
        # 获取三个关节点的坐标
        p1 = np.array([landmarks[joint_idxs[0]].x, landmarks[joint_idxs[0]].y])
        p2 = np.array([landmarks[joint_idxs[1]].x, landmarks[joint_idxs[1]].y])
        p3 = np.array([landmarks[joint_idxs[2]].x, landmarks[joint_idxs[2]].y])
        
        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算夹角（余弦定理）
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差导致超出范围
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def is_finger_extended(self, landmarks, joint_idxs):
        """判断手指是否伸直，使用角度判断，不受方向影响"""
        angle = self.calculate_finger_angle(landmarks, joint_idxs)
        return angle > 160

    def is_thumbs_up(self, landmarks, ref_length):
        """
        【精准点赞手势判断】
        只有 竖直向上竖起大拇指 + 其余四指完全弯曲 才判定为点赞
        彻底杜绝侧方、平放误触
        """
        # 1. 拇指关节点
        thumb_cmc = landmarks[1]   # 拇指根部
        thumb_tip = landmarks[4]  # 拇指尖
        
        # 2. 计算拇指方向向量（必须向上）
        dx = thumb_tip.x - thumb_cmc.x
        dy = thumb_cmc.y - thumb_tip.y  # 屏幕坐标y向下，取反
        
        # 3. 归一化判断：竖直方向分量 >> 水平分量 = 真正向上点赞
        vertical = abs(dy) / ref_length
        horizontal = abs(dx) / ref_length
        
        # 4. 严格条件：拇指向上伸直 + 其余四指完全弯曲
        thumb_upright = vertical > 0.15 and horizontal < 0.12
        other_fingers_fisted = (
            not self.is_finger_extended(landmarks, self.FINGER_JOINTS['index']) and
            not self.is_finger_extended(landmarks, self.FINGER_JOINTS['middle']) and
            not self.is_finger_extended(landmarks, self.FINGER_JOINTS['ring']) and
            not self.is_finger_extended(landmarks, self.FINGER_JOINTS['pinky'])
        )
        
        return thumb_upright and other_fingers_fisted
    
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        
        # 设置摄像头分辨率，提高处理速度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    # 识别手势并处理动作，内部已包含触发逻辑
                    self.recognize_gesture(hand_landmarks)
            
            # 转换为Qt显示格式
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            self.change_pixmap_signal.emit(convert_to_qt_format)
        
        cap.release()
    
    def recognize_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        # 1. 计算参考长度，用于距离归一化
        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        ref_length = np.linalg.norm(
            np.array([wrist.x, wrist.y]) - 
            np.array([mid_mcp.x, mid_mcp.y])
        )
        
        # 2. 判断每个手指的伸直状态
        thumb_extended = self.is_finger_extended(landmarks, self.FINGER_JOINTS['thumb'])
        index_extended = self.is_finger_extended(landmarks, self.FINGER_JOINTS['index'])
        middle_extended = self.is_finger_extended(landmarks, self.FINGER_JOINTS['middle'])
        ring_extended = self.is_finger_extended(landmarks, self.FINGER_JOINTS['ring'])
        pinky_extended = self.is_finger_extended(landmarks, self.FINGER_JOINTS['pinky'])
        
        raw_gesture = ""
        
        # 3. 手势分类
        # 握拳：所有手指弯曲，内部重置状态，但对外不显示该手势（检测不到）
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            raw_gesture = ""
        
        # 【优化】标准竖直大拇指点赞（只有这个姿势才点赞）
        elif self.is_thumbs_up(landmarks, ref_length):
            raw_gesture = "大拇指"
        
        # 仅食指伸直：1根手指，触发下一个视频
        elif index_extended and not thumb_extended and not middle_extended and not ring_extended and not pinky_extended:
            raw_gesture = "一根手指"
        
        # 食指+中指伸直：2根手指，触发上一个视频
        elif index_extended and middle_extended and not thumb_extended and not ring_extended and not pinky_extended:
            raw_gesture = "两根手指"
        
        # 手掌张开：所有手指伸直（暂停/播放）
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            raw_gesture = "手掌张开"
        
        # 其他未定义手势：对外显示无检测到
        else:
            raw_gesture = ""
        
        # 4. 防抖处理：多数投票，扩大历史窗口过滤波动
        self.gesture_history.append(raw_gesture)
        if len(self.gesture_history) >= 5:
            counter = Counter(self.gesture_history)
            valid_gesture = counter.most_common(1)[0][0]
        else:
            valid_gesture = raw_gesture
        
        # 5. 边缘触发处理，增加空帧防抖
        if valid_gesture != "":
            # 有效手势：清空空帧计数
            self.empty_frame_count = 0
            
            # 仅当手势真正变化时，才执行一次动作
            if valid_gesture != self.last_valid_gesture:
                self.gesture_signal.emit(valid_gesture)
                self.perform_action(valid_gesture)
                self.last_valid_gesture = valid_gesture
        else:
            # 空手势：计数，过滤偶尔的检测波动
            self.empty_frame_count += 1
            
            # 只有连续多帧空，才真正重置状态
            if self.empty_frame_count >= self.EMPTY_THRESHOLD:
                if self.last_valid_gesture != "":
                    self.gesture_signal.emit("")
                    self.last_valid_gesture = ""
    
    def perform_action(self, gesture):
        """执行对应手势的系统操作"""
        if gesture == "一根手指":
            pyautogui.scroll(-100)  # 下一个视频
        elif gesture == "两根手指":
            pyautogui.scroll(100)    # 上一个视频
        elif gesture == "大拇指":
            pyautogui.doubleClick()  # 点赞
        elif gesture == "手掌张开":
            pyautogui.press('space') # 暂停/播放
    
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势刷视频by1c")
        self.setGeometry(100, 100, 1000, 600)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 视频画面
        self.video_label = QLabel()
        self.video_label.setMinimumSize(400, 300)
        main_layout.addWidget(self.video_label, 3)
        
        # 控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        title_label = QLabel("手势操作说明")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        control_layout.addWidget(title_label)
        
        # 最新手势说明
        gestures = [
            ("大拇指", "竖起点赞"),
            ("一根手指", "下一个视频"),
            ("两根手指", "上一个视频"),
            ("手掌张开", "暂停/播放")
        ]
        
        self.gesture_labels = []
        for gesture, action in gestures:
            label = QLabel(f"✅ {gesture}      {action}")
            label.setStyleSheet("font-size: 14px; color: #FFFFFF;")
            self.gesture_labels.append((gesture, label))
            control_layout.addWidget(label)
        
        # 状态
        self.status_label = QLabel("当前状态：等待启动")
        self.status_label.setStyleSheet("font-size: 14px; color: #FFFFFF; margin-top: 20px;")
        control_layout.addWidget(self.status_label)
        
        self.current_gesture_label = QLabel("当前手势：无")
        self.current_gesture_label.setStyleSheet("font-size: 14px; color: #FFFFFF;")
        control_layout.addWidget(self.current_gesture_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始运行")
        self.start_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; border-radius: 5px;")
        self.start_button.clicked.connect(self.start_recognition)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止运行")
        self.stop_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #f44336; color: white; border-radius: 5px;")
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        control_layout.addLayout(button_layout)
        control_panel.setLayout(control_layout)
        control_panel.setStyleSheet("background-color: #2E2E2E; padding: 20px;")
        control_panel.setMinimumWidth(250)
        main_layout.addWidget(control_panel, 1)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: #1E1E1E;")
        
        # 线程
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.gesture_signal.connect(self.update_gesture)
    
    def start_recognition(self):
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("当前状态：运行中")
    
    def stop_recognition(self):
        self.thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("当前状态：已停止")
        self.current_gesture_label.setText("当前手势：无")
        self.video_label.clear()
    
    def update_image(self, qt_image):
        if not self.video_label.isVisible():
            return
        # 缩放图像以适应标签大小，保持宽高比
        scaled_image = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def update_gesture(self, gesture):
        # 处理空手势，显示为无
        if gesture == "":
            self.current_gesture_label.setText("当前手势：无")
        else:
            self.current_gesture_label.setText(f"当前手势：{gesture}")
        
        # 更新操作说明的高亮状态
        for gesture_name, label in self.gesture_labels:
            if gesture_name == gesture:
                label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
            else:
                label.setStyleSheet("font-size: 14px; color: #FFFFFF;")
    
    def resizeEvent(self, event):
        # 窗口大小改变时，重新调整视频显示
        super().resizeEvent(event)
        # 触发一次图像更新，确保视频画面适应新大小
        if hasattr(self, 'thread') and self.thread.running:
            # 这里不需要额外操作，因为视频线程会持续发送新帧
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
