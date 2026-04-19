# 隔空刷视频 - GestureScroll

[![GitHub Stars](https://img.shields.io/github/stars/username/gesture-control-douyin.svg)](https://github.com/username/gesture-control-douyin)
[![GitHub License](https://img.shields.io/github/license/username/gesture-control-douyin.svg)](https://github.com/username/gesture-control-douyin/blob/main/LICENSE)

你是否还在为刷视频还需要拿起手机或者电脑，下滑屏幕或者敲击键盘的↓键而烦恼，使用这个基于MediaPipe的简单小项目即可实现隔空刷视频，甚至可以躺在床上，仅仅移动手指。当然如果这样还是太累，用面部表情控制来实现同样可以，本项目使用trae辅助开发，正在开发其他端/Android/iOS

github.com/onehouzii/-GestureScroll

一个基于 MediaPipe 和 PyQt5 的手势控制工具，让你无需触摸屏幕，通过手势即可控制抖音视频的播放、暂停、上下滑动和点赞操作。

## 🚀 功能特点

- **隔空操作**：无需触摸屏幕，通过手势控制视频
- **精准识别**：基于 MediaPipe 手部关键点检测，识别准确率高
- **自适应**：基于手掌真实大小归一化，适配所有手型和距离
- **低延迟**：实时手势检测，响应迅速无卡顿
- **多手势支持**：支持上下滑动、点赞、暂停/播放等操作
- **防抖处理**：采用多数投票算法，消除摄像头抖动和检测误差

## 🛠️ 依赖框架

| 依赖项       | 版本      | 用途      |
| --------- | ------- | ------- |
| MediaPipe | ^0.10.0 | 手部关键点检测 |
| PyQt5     | ^5.15.0 | 图形界面    |
| OpenCV    | ^4.0.0  | 视频采集和处理 |
| NumPy     | ^1.20.0 | 数值计算    |
| PyAutoGUI | ^0.9.50 | 系统操作模拟  |

## 📱 手势操作说明

| 手势    | 操作     | 说明              |
| ----- | ------ | --------------- |
| 竖直大拇指 | 双击点赞   | 只有竖直向上竖起大拇指才会触发 |
| 食指下滑  | 刷下一个视频 | 仅食指伸直，向下滑动      |
| 食指上滑  | 刷上一个视频 | 仅食指伸直，向上滑动      |
| 手掌张开  | 暂停/播放  | 所有手指伸直          |
| 握拳    | 重置状态   | 所有手指弯曲，清空缓存     |

## 🧠 核心算法逻辑

### 1. 手掌真实大小归一化算法

**作用**：消除手型大小、摄像头距离对滑动控制的影响，实现全场景手感统一

**选取基准点**：手腕节点 (0)、中指根部节点 (9)

**计算公式**：

```python
ref_length = np.linalg.norm(
    np.array([wrist.x, wrist.y]) - 
    np.array([mid_mcp.x, mid_mcp.y])
)
```

**应用场景**：所有食指滑动位移计算

### 2. 食指加权位移上下滑控制算法

**触发条件**：仅食指伸直，其余 4 指弯曲

**加权坐标计算**：

```python
# 节点顺序：5(根部0.1) → 6(近端0.1) →7(中端0.3)→8(指尖0.6)
index_5 = landmarks[5]
index_6 = landmarks[6]
index_7 = landmarks[7]
index_8 = landmarks[8]

# 加权计算：5号0.1，6号0.1，7号0.3，8号0.6
weighted_y = 0.1 * index_5.y + 0.1 * index_6.y + 0.3 * index_7.y + 0.6 * index_8.y
```

**位移计算**：

- 缓存连续 10 帧加权 Y 坐标
- 总位移 = (最后一帧 Y 坐标 - 第一帧 Y 坐标)
- 归一化总位移 = 总位移 / 手掌基准长度

**动作判定**：

- 归一化位移 > 0.15 → 执行食指下滑
- 归一化位移 < -0.15 → 执行食指上滑
- 触发后清空缓存，避免连续触发

### 3. 手指伸直/弯曲状态判断算法

**作用**：统一判定所有手指的伸直/弯曲状态，为手势识别提供基础

**计算逻辑**：余弦定理计算关节夹角

```python
def calculate_finger_angle(self, landmarks, joint_idxs):
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
```

**判定规则**：

- 夹角 ≥ 160° → 手指伸直
- 夹角 < 160° → 手指弯曲

### 4. 大拇指点赞手势识别算法

**作用**：精准识别竖直点赞手势，杜绝误触

**基础条件**：大拇指伸直，其余四指完全弯曲

**方向判定**：

```python
def is_thumbs_up(self, landmarks, ref_length):
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
```

**严格阈值**：

- 垂直分量 > 0.15 且 水平分量 < 0.12
- 满足所有条件 → 判定为大拇指点赞

### 5. 手掌张开暂停/播放算法

**触发条件**：拇指、食指、中指、无名指、小指 全部伸直

**执行动作**：模拟键盘空格键

### 6. 握拳状态重置算法

**触发条件**：所有手指 全部弯曲

**执行操作**：

- 清空食指滑动缓存
- 重置手势状态
- 无系统操作

### 7. 手势防抖 + 边缘触发算法

**作用**：消除摄像头抖动、检测误差，保证手势操作稳定

**防抖逻辑（多数投票）**：

```python
# 防抖处理：多数投票
self.gesture_history.append(raw_gesture)
if len(self.gesture_history) >= 3:
    counter = Counter(self.gesture_history)
    valid_gesture = counter.most_common(1)[0][0]
else:
    valid_gesture = raw_gesture
```

**边缘触发逻辑**：

```python
# 边缘触发：仅手势变化时执行一次
if valid_gesture != self.last_valid_gesture and valid_gesture != "" and valid_gesture != "握拳":
    self.gesture_signal.emit(valid_gesture)
    self.perform_action(valid_gesture)
    self.last_valid_gesture = valid_gesture
```

## 🔧 安装和使用

### 安装依赖

```bash
pip install mediapipe opencv-python pyqt5 numpy pyautogui
```

### 运行程序

```bash
python gesture_control.py
```

### 使用步骤

1. 运行程序后，点击"开始运行"按钮
2. 将手放在摄像头前，确保手部清晰可见
3. 使用以下手势控制视频：
   - 竖起大拇指：双击点赞
   - 仅伸直食指并上下移动：上下滑动视频
   - 张开手掌：暂停/播放视频
   - 握拳：重置状态
4. 点击"停止运行"按钮结束程序

## 🎯 算法核心特性

1. **自适应**：基于手掌真实大小归一化，适配所有手型、距离
2. **精准加权**：食指指尖主导位移，根节点低权重参与，符合人体控制习惯
3. **高鲁棒性**：角度判定 + 方向判定 + 防抖三重过滤，无误触
4. **低延迟**：帧间实时计算，滑动响应无卡顿

## 📁 项目结构

```
├── gesture_control.py  # 主程序文件
├── README.md          # 项目说明文档
├── 技术方案.md         # 技术方案文档
└── 算法实现原理.md       # 算法实现原理文档
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🌟 致谢

- [MediaPipe](https://mediapipe.dev/) - 提供手部关键点检测
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - 提供图形界面
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - 提供系统操作模拟

***

**享受隔空刷视频的乐趣！** 🎵✨
