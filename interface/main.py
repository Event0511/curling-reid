import os
import sys
import cv2
import time
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel, \
    QStatusBar, QToolBar, QApplication
from PyQt5.uic import loadUi

from pose.custom.camera import Camera
from pose.custom.label_frame import LabelFrame
from pose.custom.dock_filetree import FiletreeDock
from pose.custom.dock_setting import SettingDock
from pose.custom.openpose_model import OpenposeModel
from pose.custom.save_widget import SaveWidget
from pose.custom.gesture_model import GestureModel
from pose.custom.dock_media import MediaDock

from interface.module.presenter import Presenter


class OpenposeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("ui/main_window.ui", self)
        self.setWindowTitle("OpencvGUI")
        self.setWindowIcon(QIcon("icon/logo.png"))
        self.reid_model_path = '../reid/models/stageIII/best_net_E.pth'
        self.gesture_model_path = r'../pose/models/gesture/model@acc0.992.pth'

        # 工具栏
        # 所有动作定义在ui文件中
        self.tool_bar = QToolBar()
        self.tool_bar.addAction(self.action_camera)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.action_save)
        self.tool_bar.addAction(self.action_autosave)
        self.tool_bar.addSeparator()
        self.tool_bar.addAction(self.action_setting)
        self.tool_bar.addAction(self.action_filetree)
        self.addToolBar(self.tool_bar)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_fps = QLabel("FPS:00.0")
        self.status_bar.addPermanentWidget(self.status_fps)
        self.setStatusBar(self.status_bar)

        # 组件
        self.label_frame = LabelFrame(self)  # 用于显示画面

        self.dock_setting = SettingDock(self)  # 设置
        self.dock_filetree = FiletreeDock(self)  # 目录
        self.dock_media = MediaDock(self)  # 视频播放控制

        self.timer = QTimer()
        self.camera = Camera()  # 视频控制器
        self.openpose_model = OpenposeModel()  # openpose模型
        self.save_widget = SaveWidget()  # 保存窗口
        self.gesture_model = GestureModel(self.gesture_model_path)

        # self.person_tracker = PersonTracker()  # 加载person tracker
        self.controller = Presenter(self.reid_model_path, 0.95)  # reid的presenter
        self.out = None

        # 设置dock
        self.setCentralWidget(self.label_frame)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_setting)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_filetree)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_media)

        # 工具栏动作
        self.action_camera.triggered.connect(self.run_camera)
        self.action_filetree.triggered.connect(self.show_filetree)
        self.action_setting.triggered.connect(self.show_setting)
        self.action_save.triggered.connect(self.save)
        self.action_autosave.triggered.connect(self.auto_save)

        # 相机计时器
        self.camera.timer.timeout.connect(self.update_frame)
        # 自动保存计时器
        self.timer.timeout.connect(self.save)

    def update_frame(self, interval=2):
        """更新frame画面"""
        start_time = time.time()
        self.dock_media.update_slider(self.frame_pos, self.frame_count)
        frame = self.frame
        if frame is None:
            return None
        # result, ids, keypoints = self.openpose_model(frame)
        # 为提升视频流畅度，每interval帧输入一张图片，在函数输入处设定，默认为2
        if int(self.frame_pos - 1) % interval == 0:
            self.controller.update_people(frame)
        result = self.controller.draw_rects(frame)
        self.out.write(result)
        # result, keypoints = self.gesture_recognition(result, keypoints)
        #
        # message = self.generate_message(keypoints)
        # result, obj_ids, pos, imgs = self.controller.tracker.track(frame)

        self.label_frame.update_frame(result)
        fps = 1 / (time.time() - start_time)
        if int(self.frame_pos - 1) % interval == 0:
            self.status_fps.setText("FPS:{:.1f}".format(fps))
        # self.status_bar.showMessage(message)
        # return result, keypoints
        return result

    """custom"""
    def show_rect(self, result, keypoints):
        if keypoints[0].size == 1:
            return
        bodys = keypoints[0]
        if bodys[0].size >= 1:
            person_num = bodys.shape[0]
            for i in range(person_num):  # 每个人
                for body in bodys:  # 每只手
                    rect = self.openpose_model.body_bbox(body)
                    if rect:
                        x, y, w, h = rect
                        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        # cv2.putText(result, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return result, keypoints
    """custom"""

    def save(self):
        """执行保存操作"""
        print("Video saved.")
        self.out.release()
        # 原先程序的保存是保存当前帧的姿态检测结果
        # if not self.label_frame.pixmap():
        #     QMessageBox.warning(self, "Note", "No data in frame", QMessageBox.Yes)
        #     return

        # pixmap = self.label_frame.img_to_pixmap(self.openpose_model.get_rendered_image())
        # body, hand, face = self.openpose_model.get_keypoints()
        # message = self.generate_message((body, hand, face))
        # body = copy.deepcopy(body) if self.body_on else None
        # hand = copy.deepcopy(hand) if self.hand_on else None
        # face = copy.deepcopy(face) if self.face_on else None
        # keypoints = (body, hand, face)
        #
        # if self.timer.isActive():
        #     self.save_widget.save(pixmap.copy(), *keypoints)
        # else:
        #     self.save_widget.set_frame(pixmap.copy(), *keypoints, message)
        #     self.save_widget.show()

    def auto_save(self):
        """开启或关闭自动保存"""
        if not self.camera.is_open:
            self.action_autosave.setChecked(False)
        if self.action_autosave.isChecked():
            self.timer.start(self.save_interval * 1000)
        else:
            self.timer.stop()

    # 功能
    def show_setting(self):
        """隐藏或显示设置"""
        if self.dock_setting.isHidden():
            self.dock_setting.show()
        else:
            self.dock_setting.hide()

    def show_filetree(self):
        """隐藏或显示目录树"""
        if self.dock_filetree.isHidden():
            self.dock_filetree.show()
        else:
            self.dock_filetree.hide()

    def run_image(self, img_path):
        """图片"""
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if image is None:
            QMessageBox.warning(self, "Note", "Read Image error", QMessageBox.Yes)
            return
        result, ids, keypoints = self.openpose_model(image)
        result = self.draw_rects(result)
        message = self.generate_message(keypoints)
        self.label_frame.update_frame(result)
        self.status_bar.showMessage(message)
        self.dock_media.hide()

    def run_video(self, video_path):
        print('视频')
        """视频"""
        self.controller = Presenter(self.reid_model_path, 0.95)  # reid的presenter
        self.camera.start(video_path)
        self.label_frame.resize(*self.resolution)

        video_w = self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_h = self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = self.camera.cap.get(cv2.CAP_PROP_FPS)
        video_name = os.path.basename(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output/output_' + video_name, fourcc=fourcc, fps=int(video_fps),
                                   frameSize=(int(video_w), int(video_h)))
        self.update_frame()  # 初始化画面
        self.camera.pause()
        self.dock_media.reset()
        self.dock_media.show()

    def run_camera(self):
        """相机"""
        if self.action_camera.isChecked():
            self.camera.start(0)
            self.label_frame.resize(*self.resolution)
        else:
            self.label_frame.clear()
            self.camera.stop()
            self.status_fps.setText("FPS:00.0")
        self.dock_media.hide()

    def gesture_recognition(self, result, keypoints):
        """实现手势识别"""
        if self.gesture_on:
            hands = keypoints[1]
            if hands[0].size == 1:
                return result, keypoints
            person_num = hands[0].shape[0]
            for i in range(person_num):  # 每个人
                for hand in hands:  # 每只手
                    rect = self.gesture_model.hand_bbox(hand[i])
                    gesture = self.gesture_model(hand[i])
                    if rect:
                        print(rect)
                        x, y, w, h = rect
                        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        cv2.putText(result, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return result, keypoints

    def generate_message(self, keypoints):
        """获取识别结果信息"""
        if keypoints[0].size != 1:
            person_num = keypoints[0].shape[0]
            message = "person: {}".format(person_num)
            for i in range(person_num):
                message += " | person{}(".format(i + 1)
                if self.body_on:
                    pose_keypoints = keypoints[0][i, :, 2]
                    pose_detected = pose_keypoints[pose_keypoints > self.body_threshold]
                    message += "pose: {:>2d}/{}".format(pose_detected.size, 25)

                if self.hand_on:
                    right_hand_keypoinys = keypoints[1][0][i, :, 2]
                    left_hand_keypoinys = keypoints[1][1][i, :, 2]

                    right_hand_detected = right_hand_keypoinys[
                        right_hand_keypoinys > self.hand_threshold]
                    left_hand_detected = left_hand_keypoinys[left_hand_keypoinys > self.hand_threshold]
                    message += " | right hand: {:>2d}/{}".format(right_hand_detected.size, 21)
                    message += " | left hand: {:>2d}/{}".format(left_hand_detected.size, 21)
                message += ")"
        else:
            message = "person: {}".format(0)
        return message

    @property
    def body_on(self):
        return self.dock_setting.body_on

    @property
    def hand_on(self):
        return self.dock_setting.hand_on

    @property
    def face_on(self):
        return self.dock_setting.face_on

    @property
    def body_threshold(self):
        return self.dock_setting.body_threshold

    @property
    def hand_threshold(self):
        return self.dock_setting.hand_threshold

    @property
    def face_threshold(self):
        return self.dock_setting.face_threshold

    @property
    def gesture_on(self):
        return self.dock_setting.gesture_on

    @property
    def save_interval(self):
        return self.dock_setting.save_interval

    @property
    def frame(self):
        return self.camera.frame

    @property
    def frame_pos(self):
        return self.camera.frame_pos

    @property
    def frame_count(self):
        return self.camera.frame_count

    @property
    def resolution(self):
        return self.camera.resolution


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpenposeGUI()
    window.show()
    sys.exit(app.exec_())
