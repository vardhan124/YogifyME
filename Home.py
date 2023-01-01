
import streamlit as st
from pyvis.network import Network
from PIL import Image
import cv2
import copy
import mediapipe as mp
import numpy as np
from playsound import playsound
import pandas as pd
stage = 12
from streamlit_lottie import st_lottie

def extractKeypoint(path):
    IMAGE_FILES = [path]
    stage = None
    joint_list_video = pd.DataFrame([])
    count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_h, image_w, _ = image.shape

            try:

                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                joints = []
                joint_list = pd.DataFrame([])

                for i, data_point in zip(range(len(landmarks)), landmarks):
                    joints = pd.DataFrame({
                        'frame': count,
                        'id': i,
                        'x': data_point.x,
                        'y': data_point.y,
                        'z': data_point.z,
                        'vis': data_point.visibility
                    }, index=[0])
                    joint_list = joint_list.append(joints, ignore_index=True)

                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                    })

                angle = []
                angle_list = pd.DataFrame([])
                angle1 = angle_cal(right_shoulder, right_elbow, right_wrist)
                angle.append(int(angle1))
                angle2 = angle_cal(left_shoulder, left_elbow, left_wrist)
                angle.append(int(angle2))
                angle3 = angle_cal(right_elbow, right_shoulder, right_hip)
                angle.append(int(angle3))
                angle4 = angle_cal(left_elbow, left_shoulder, left_hip)
                angle.append(int(angle4))
                angle5 = angle_cal(right_shoulder, right_hip, right_knee)
                angle.append(int(angle5))
                angle6 = angle_cal(left_shoulder, left_hip, left_knee)
                angle.append(int(angle6))
                angle7 = angle_cal(right_hip, right_knee, right_ankle)
                angle.append(int(angle7))
                angle8 = angle_cal(left_hip, left_knee, left_ankle)
                angle.append(int(angle8))

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(0, 0, 255), thickness=4, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(0, 255, 0), thickness=4, circle_radius=2)

                                          )
                image = cv2.resize(image, (640, 480))

            except:
                pass
            joint_list_video = joint_list_video.append(
                joint_list, ignore_index=True)
            #cv2.rectangle(image,(0,0), (100,255), (255,255,255), -1)

            #cv2.putText(image, 'ID', (10,14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,255], 2, cv2.LINE_AA)
            #cv2.putText(image, str(1), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(2), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(3), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(4), (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(5), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(6), (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(7), (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(8), (10,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)

            #cv2.putText(image, 'Angle', (40,12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,255], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle1)), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle2)), (40,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle3)), (40,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle4)), (40,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle5)), (40,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle6)), (40,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle7)), (40,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            #cv2.putText(image, str(int(angle8)), (40,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(0, 0, 255), thickness=4, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(0, 255, 0), thickness=4, circle_radius=2)

                                      )

            #cv2.imshow('MediaPipe Feed',image)

            # if cv2.waitKey(0) & 0xFF == ord('q'):
            # break

        # cv2.destroyAllWindows()
    return landmarks, keypoints, angle, image


def match_pose(x, image, landmarks, keypoints_list, angles_list, frames, stage):
    names = ["Pranamasana", "Hasta Uttanasana", "Pada Hastasana", "Ashwa Sanchalanasana", "Dandasana", "Astanga Namaskara","Bhujangasana", "Adho Mukha Svanasana", "Ashwa Sanchalanasana", 'Pada Hastasana', "Uttana Hastasana", "Pranamasana"]
                
    
    target_cords = x[1]
    target_angles = x[2]
    flag = 0
    
    height,width,depth = image.shape
    if angles_list[0]-target_angles[0] >=30:         #right elbow angle
        flag = 1
        with error_c.container():
            st.warning("Fold arm at right elbow")
        cv2.circle(image,(int(keypoints_list[0][0]*width), int(keypoints_list[0][1]*height)),30,(0,0,255),5) 
    
    if angles_list[0]-target_angles[0] <-30:
        flag = 1
        with error_c.container():
            st.warning("Extend arm at right elbow")
        cv2.circle(image,(int(keypoints_list[0][0]*width), int(keypoints_list[0][1]*height)),30,(0,0,255),5) 
    
    if angles_list[1]-target_angles[1] <-30:          #left elbow angle
        flag =1
        with error_c.container():
            st.warning("Extend arm at left elbow")
        cv2.circle(image,(int(keypoints_list[1][0]*width), int(keypoints_list[1][1]*height)),30,(0,0,255),5) 
    
    if angles_list[1]-target_angles[1] >30:
        flag = 1
        with error_c.container():
            st.warning("Fold arm at left elbow")
        cv2.circle(image,(int(keypoints_list[1][0]*width), int(keypoints_list[1][1]*height)),30,(0,0,255),5) 
    
    if angles_list[2]-target_angles[2] >= 30:       #right shoulder angle
        flag =1
        with error_c.container():
            st.warning("Bring down your right elbow")
        cv2.circle(image,(int(keypoints_list[2][0]*width), int(keypoints_list[2][1]*height)),30,(0,0,255),5) 
      
    if angles_list[2]-target_angles[2] <-30:
        flag = 1
        with error_c.container():
            st.warning("Bing up your right elbow")
        cv2.circle(image,(int(keypoints_list[2][0]*width), int(keypoints_list[2][1]*height)),30,(0,0,255),5) 
    
    if angles_list[3]-target_angles[3] >= 30:       #left shoulder angle
        flag =1
        with error_c.container():
            st.warning("Bring down your left elbow")
        cv2.circle(image,(int(keypoints_list[3][0]*width), int(keypoints_list[3][1]*height)),30,(0,0,255),5) 
      
    if angles_list[3]-target_angles[3] <-30:
        flag = 1
        with error_c.container():
            st.warning("Bring up your left elbow")
        cv2.circle(image,(int(keypoints_list[3][0]*width), int(keypoints_list[3][1]*height)),30,(0,0,255),5)
    ##
    if angles_list[4]-target_angles[4] >= 20:   #right hip    
        flag =1
        with error_c.container():
            st.warning("Lean forward")
        cv2.circle(image,(int(keypoints_list[4][0]*width), int(keypoints_list[4][1]*height)),30,(0,0,255),5) 
      
    if angles_list[4]-target_angles[4] <-20:   
        flag = 1
        
        cv2.putText(image, str(""), (10,320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,0,255], 2, cv2.LINE_AA)
        cv2.circle(image,(int(keypoints_list[4][0]*width), int(keypoints_list[4][1]*height)),30,(0,0,255),5)
    if angles_list[5]-target_angles[5] >= 20:       #left hip
        flag =1
        cv2.putText(image, str(""), (10,340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,0,255], 2, cv2.LINE_AA)
        cv2.circle(image,(int(keypoints_list[5][0]*width), int(keypoints_list[5][1]*height)),30,(0,0,255),5) 
      
    if angles_list[5]-target_angles[5] <-20:
        flag = 1
        with error_c.container():
            st.warning("Lean backward")
        cv2.circle(image,(int(keypoints_list[5][0]*width), int(keypoints_list[5][1]*height)),30,(0,0,255),5)
   
    if angles_list[6]-target_angles[6] >= 30:       #right knee
        flag =1
        with error_c.container():
            st.warning("Bend your right knee")
        cv2.circle(image,(int(keypoints_list[6][0]*width), int(keypoints_list[6][1]*height)),30,(0,0,255),5) 
      
    if angles_list[6]-target_angles[6] <-30:
        flag = 1
        with error_c.container():
            st.warning("Extend leg at right knee")
        cv2.circle(image,(int(keypoints_list[6][0]*width), int(keypoints_list[6][1]*height)),30,(0,0,255),5)

    if angles_list[7]-target_angles[7] >= 30:       #left knee
        flag =1
        with error_c.container():
            st.warning("Bend your left knee")
        cv2.circle(image,(int(keypoints_list[7][0]*width), int(keypoints_list[7][1]*height)),30,(0,0,255),5) 
      
    if angles_list[7]-target_angles[7] <-30:
        flag = 1
        with error_c.container():
            st.warning("Extend leg at left knee")
        cv2.circle(image,(int(keypoints_list[7][0]*width), int(keypoints_list[7][1]*height)),30,(0,0,255),5)
   
      
      
    if flag == 1:
        with a_c.container():
         st.info(str(stage)+' '+names[stage-1])
    else:
        with error_c.container():
            st.success("CORRECT POSTURE")
        return frames +1
    return 0

def angle_cal(a, b, c):
    a = np.array(a)  # end
    b = np.array(b)  # center
    c = np.array(c)  # end

    temp = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    temp = np.abs(temp*180.0)/(np.pi)

    if (temp >= 180):
        return round(360 - temp)
    return round(temp)


st.set_page_config(
    page_title="Yoga Pose Correction",
    page_icon="🖥️",
)

def add_bg_from_url(url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({url});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url("https://img.freepik.com/free-vector/elegant-white-background-with-shiny-lines_1017-17580.jpg?w=996&t=st=1672557975~exp=1672558575~hmac=d8e8a14313ee88b5cb1b06549b801805b39f6a564488b4408efa9e32809ff10e") 
images = ["stage1.jpg", "stage2.jpg", "stage3.jpg", "stage4.png",
          "stage5.png", "stage6.jpg", "stage7.jpg", "stage8.png"]
asanas_names = ["Pranamasana", "Hasta Uttanasana", "Pada Hastasana", "Ashwa Sanchalanasana", "Dandasana", "Astanga Namaskara",
                "Bhujangasana", "Adho Mukha Svanasana", "Ashwa Sanchalanasana", 'Pada Hastasana', "Uttana Hastasana", "Pranamasana","Benifits of suryanamaskara"]

import requests
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://assets6.lottiefiles.com/packages/lf20_fbpbl0qw.json"
#lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)

c_1,c_2=st.columns(2)
with c_1:
    #st_lottie(lottie_hello, key="hello",height=250,width=200)
    pass
with c_2:
    st.title("YogifyMe")
    st.write("Yoga Pose Correction  !!!")
t_c=st.empty()
co_1, col_2 = st.columns(2)
with co_1:
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    img_c = st.empty()


col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12)

cols = [col1, col2, col3, col4, col5, col6,
        col7, col8, col9, col10, col11, col12]
for i in range(12):
            with cols[i]:
                
                    st.text(i+1)
with col1:
    col1_c = st.empty()
with col2:
    col2_c = st.empty()
with col3:
    col3_c = st.empty()
with col4:
    col4_c = st.empty()
with col5:
    col5_c = st.empty()
with col6:
    col6_c = st.empty()
with col7:
    col7_c = st.empty()
with col8:
    col8_c = st.empty()
with col9:
    col9_c = st.empty()
with col10:
    col10_c = st.empty()
with col11:
    col11_c = st.empty()
with col12:
    col12_c = st.empty()
cols_c=[col1_c,col2_c,col3_c,col4_c,col5_c,col6_c,col7_c,col8_c,col9_c,col10_c,col11_c,col12_c]


for i in range(stage-2):
    
    with cols[i]:

                with cols_c[i].container():
                    
                    st.text("✔️")
with cols[stage-1]:
            with cols_c[stage-1].container():
                st.text("❓")

with col_2:
    a_c=st.empty()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    paths = ["stage1.jpg", "stage2.jpg", "stage3new.jpeg", "stage4.png", "stage5.png", "stage6.jpg",
             "stage7.jpg", "stage8.png", "stage4.png", "stage3new.jpeg", "stage2.jpg", "stage1.jpg","benifits.png"]
    frame=st.empty()
    with frame.container():         
        FRAME_WINDOW = st.image([])
    error_c = st.empty()
    cap = cv2.VideoCapture(0)
    
    while(stage <= 13):
        if stage!=1:
            with cols[stage-2]:
                
                with cols_c[stage-2].container():
                    st.text("✔️")
        if stage!=13:
            with cols[stage-1]:
                with cols_c[stage-1].container():
                    st.text("❓")
        if stage==13:
            with col_2:
                
                frame.container().empty()
                error_c.container().empty()
            with co_1:
                    a_c.container().empty()
                    img_c.container().empty()
            with t_c.container():
                image = Image.open("asanas/"+paths[stage-1])
                new_image = image.resize((400, 400))
                st.info("You successfully Finished 12 Steps of SuryaNamaskara ")
                st.image(new_image, caption=asanas_names[stage-1])
            cap.release()

            break
        
        x = extractKeypoint("asanas/"+paths[stage-1])
        frames = 0
        with co_1:
            with img_c.container():
                
                    image = Image.open("asanas/"+paths[stage-1])
                    new_image = image.resize((400, 300))
                
                    st.image(new_image, caption=asanas_names[stage-1])
        
        
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                # recolor to RGB
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # detection making
                results = pose.process(image)
                # draw pose annotation on image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                keypoints_list = []
                angles_list = []

                try:
                    # storing coordinates
                    landmarks = results.pose_landmarks.landmark

                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    keypoints_list.append(right_elbow)

                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    keypoints_list.append(left_elbow)

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    keypoints_list.append(right_shoulder)

                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    keypoints_list.append(left_shoulder)

                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    keypoints_list.append(right_hip)

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    keypoints_list.append(left_hip)

                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    keypoints_list.append(right_knee)

                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    keypoints_list.append(left_knee)

                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    keypoints_list.append(left_wrist)

                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    keypoints_list.append(right_wrist)

                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    keypoints_list.append(left_ankle)

                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    keypoints_list.append(right_ankle)

                    # calculate angle
                    angle2 = angle_cal(
                        right_shoulder, right_elbow, right_wrist)
                    angles_list.append(angle2)

                    angle1 = angle_cal(left_shoulder, left_elbow, left_wrist)
                    angles_list.append(angle1)

                    angle4 = angle_cal(right_elbow, right_shoulder, right_hip)
                    angles_list.append(angle4)

                    angle3 = angle_cal(left_elbow, left_shoulder, left_hip)
                    angles_list.append(angle3)

                    angle5 = angle_cal(right_shoulder, right_hip, right_knee)
                    angles_list.append(angle5)

                    angle6 = angle_cal(left_shoulder, left_hip, left_knee)
                    angles_list.append(angle6)

                    angle7 = angle_cal(right_hip, right_knee, right_ankle)
                    angles_list.append(angle7)

                    angle8 = angle_cal(left_hip, left_knee, left_ankle)
                    angles_list.append(angle8)

                    # output on screen
                    cv2.putText(image, str(angle1), tuple(np.multiply(left_elbow, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle2), tuple(np.multiply(right_elbow, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle3), tuple(np.multiply(left_shoulder, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle4), tuple(np.multiply(right_shoulder, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle5), tuple(np.multiply(right_hip, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle6), tuple(np.multiply(right_knee, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(angle7), tuple(np.multiply(left_knee, [640, 480]).astype(
                        int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    frames = match_pose(
                        x, image, landmarks, keypoints_list, angles_list, frames, stage)
                    if frames >= 30:

                        # stage=st.session_state("stage")+1
                        stage += 1
                        
                        break

                except:
                    print("failed")

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()



                )

            # flip image
            #     image = cv2.flip(image,1)
                
                FRAME_WINDOW.image(image,channels = 'BGR')

               
    cap.release()
    
    
    cv2.destroyAllWindows()
