import mediapipe as mp
import cv2
import cv2 as cv; cv2=cv
import numpy as np
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    angle = int(angle)

    return angle

def calculate_back_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    angle = int(angle)

    return angle

def calculate_length(a, b):
    a = np.array(a)
    b = np.array(b)
    p1 = (a[0] - b[0]) ** 2
    p2 = (a[1] - b[1]) ** 2
    length = np.sqrt(p1+p2)
    return length

def calculate_slope(a, b):
    slope = (b[1]-a[1])/(b[0]-a[0])
    return slope

def calculate_midpoint(a, b):
    midpoint = [(a[0]+b[0])/2, (a[1]+b[1])/2]
    return midpoint

counter = 0
stage = None

def calculate_xy(a):
    a = np.array(a)
    return a

cap = cv2.VideoCapture(0)
width , height = 1280 , 720
cap.set(3, width)
cap.set(4, height)
"""ret, img = cap.read()
img2 = cv2.flip(img, 1)"""


cv2.namedWindow('Mediapipe Feed')
def nothing(x):
    print(x)
    pass

prev_frame_time = 0
new_frame_time = 0
timer = 0
timer_reset = 0
timeCount = 0
timetime = 0
back_damage_timer = 0
neck_damage_timer = 0
right_wrist_damage_timer = 0
left_wrist_damage_timer = 0

left_wrist_health = 600
right_wrist_health = 600
back_health = 600
neck_health = 600

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image,(width, height))

        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        #def on_first(*args):
        try:
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            timeCount = int(fps)



            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x, landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            zpos_leftWrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].z

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x, landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]
            right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x, landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
            zpos_rightWrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].z

            face_nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y, landmarks[mp_pose.PoseLandmark.NOSE].z]



            right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
            left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)

            back_angle = (right_back_angle + left_back_angle) / 2

            hips = calculate_midpoint(right_hip, left_hip)
            shoulders = calculate_midpoint(right_shoulder, left_shoulder)
            left_hand = calculate_midpoint(left_index, left_pinky)
            right_hand = calculate_midpoint(right_index, right_pinky)

            left_wrist_angle = calculate_angle(right_hand, right_wrist, right_elbow)
            right_wrist_angle = calculate_angle(left_hand, left_wrist, left_elbow)
            neck_angle = calculate_angle(hips, shoulders, face_nose)
            if(back_angle < 70):
                print("damage has been taken")
                back_damage_timer += 1/timeCount
                back_health -= back_damage_timer
                back_health_percentage = back_health / 600

            if(neck_angle < 130):
                print("damage has been taken")
                neck_damage_timer += 1/timeCount
                neck_health -= neck_damage_timer
                neck_health_percentage = neck_health / 600

            if(right_wrist_angle < 130):
                print("damage has been taken")
                right_wrist_damage_timer += 1/timeCount
                right_wrist_health = right_wrist_health-right_wrist_damage_timer
                right_wrist_health_percentage = right_wrist_health / 600
                if (right_wrist_health_percentage == 0):


            if(left_wrist_angle < 130):
                print("damage has been taken")
                left_wrist_damage_timer += 1/timeCount
                left_wrist_health = left_wrist_health-left_wrist_damage_timer
                left_wrist_health_percentage = left_wrist_health / 600
#chicken

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                           )
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(image.shape)
            timer=0
            break
timer=0
cap.release()
cv2.destroyAllWindows()