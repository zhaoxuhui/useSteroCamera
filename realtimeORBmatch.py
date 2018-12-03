# coding=utf-8
import cv2
import numpy as np


def drawKeyPoints(img, kps, color=(0, 0, 255), rad=3):
    if img.shape.__len__() == 2:
        img_pro = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_pro = img
    for point in kps:
        pt = (int(point.pt[0]), int(point.pt[1]))
        cv2.circle(img_pro, pt, rad, color, 1, cv2.LINE_AA)
    return img_pro


def drawMatches(img1, img2, good_matches):
    if img1.shape.__len__() == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.shape.__len__() == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img_out = np.zeros([max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3], np.uint8)
    img_out[:img1.shape[0], :img1.shape[1], :] = img1
    img_out[:img2.shape[0], img1.shape[1]:, :] = img2
    for match in good_matches:
        pt1 = (int(match[0]), int(match[1]))
        pt2 = (int(match[2] + img1.shape[1]), int(match[3]))
        cv2.circle(img_out, pt1, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img_out, pt2, 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img_out, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return img_out


def getORBkpts(img, features=1000):
    orb = cv2.ORB_create(nfeatures=features)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def matchORBkpts(kps1, desp1, kps2, desp2):
    good_kps1 = []
    good_kps2 = []
    good_matches = []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desp1, desp2)

    if matches.__len__() == 0:
        return good_kps1, good_kps2, good_matches
    else:
        min_dis = 10000
        for item in matches:
            dis = item.distance
            if dis < min_dis:
                min_dis = dis

        g_matches = []
        for match in matches:
            if match.distance <= max(1.1 * min_dis, 15.0):
                g_matches.append(match)

        for i in range(g_matches.__len__()):
            good_kps1.append([kps1[g_matches[i].queryIdx].pt[0], kps1[g_matches[i].queryIdx].pt[1]])
            good_kps2.append([kps2[g_matches[i].trainIdx].pt[0], kps2[g_matches[i].trainIdx].pt[1]])
            good_matches.append([kps1[g_matches[i].queryIdx].pt[0], kps1[g_matches[i].queryIdx].pt[1],
                                 kps2[g_matches[i].trainIdx].pt[0], kps2[g_matches[i].trainIdx].pt[1]])

        return good_kps1, good_kps2, good_matches


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

        # 获取ORB特征点
        kps_left, desc_left = getORBkpts(left_cam)
        kps_right, desc_right = getORBkpts(right_cam)
        # 左右影像特征点匹配
        good_kps1, good_kps2, good_matches = matchORBkpts(kps_left, desc_left, kps_right, desc_right)
        # 绘制匹配点对
        match_img = drawMatches(left_cam, right_cam, good_matches)
        cv2.imshow("match_img", match_img)
        cv2.waitKey(25)
