# coding=utf-8
import cv2
import sys


def getCameraInstance(index, resolution):
    """
    用于初始化获取相机实例，从而读取数据

    :param index: 相机的索引编号，如果只有一个相机那就是0，有多个则以此类推
    :param resolution: 相机数据的分辨率设置
    :return: 相机实例，以及设置的影像长宽
    """
    cap = cv2.VideoCapture(index)

    if resolution == '960p':
        width = 2560
        height = 960
    elif resolution == '480p':
        width = 1280
        height = 480
    elif resolution == '240p':
        width = 640
        height = 240

    # OpenCV有相关API可以设置视频流的长宽
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_YUYV)
    return cap, width, height


if __name__ == '__main__':

    # 读取启动参数，设置分辨率
    if sys.argv.__len__() == 3:
        cam_no = int(sys.argv[1])
        reso_flag = sys.argv[2]
    else:
        cam_no = 1
        reso_flag = '960p'

    # 获取相机实例并返回对象
    cap, width, height = getCameraInstance(cam_no, reso_flag)

    # 不断循环读取帧数据
    while True:
        ret, frame = cap.read()

        # 对影像进行拆分，左右影像
        left_cam = frame[:, :int(width / 2), :]
        right_cam = frame[:, int(width / 2):, :]

        # 分别显示
        cv2.imshow("left_cam", left_cam)
        cv2.imshow("right_cam", right_cam)
        cv2.waitKey(25)
