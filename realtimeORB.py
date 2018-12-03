# coding=utf-8
import cv2


def drawKeyPoints(img, kps, color=(0, 0, 255), rad=3):
    if img.shape.__len__() == 2:
        img_pro = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_pro = img
    for point in kps:
        pt = (int(point.pt[0]), int(point.pt[1]))
        cv2.circle(img_pro, pt, rad, color, 1, cv2.LINE_AA)
    return img_pro


def getORBkpts(img, features=1000):
    orb = cv2.ORB_create(nfeatures=features)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def getCameraInstance(index, resolution):
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, width, height


if __name__ == '__main__':

    cap, width, height = getCameraInstance(1, '480p')

    while True:
        ret, frame = cap.read()
        left_cam = frame[:, :int(width / 2), :]
        right_cam = frame[:, int(width / 2):, :]

        # ORB特征提取部分
        kps_left, desc_left = getORBkpts(left_cam)
        kps_right, desc_right = getORBkpts(right_cam)
        # 绘制ORB特征点
        left_kps = drawKeyPoints(left_cam, kps_left)
        right_kps = drawKeyPoints(right_cam, kps_right)

        cv2.imshow("left_kps", left_kps)
        cv2.imshow("right_kps", right_kps)
        cv2.waitKey(25)
