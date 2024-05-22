import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import dlib
from imutils.video import VideoStream, FPS
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib.mailer import Mailer
from mylib import config, thread
import imutils


def process_frame(frame, net, ct, trackableObjects, totalUp, totalDown):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    H, W = frame.shape[:2]

    status = "Waiting"
    rects = []

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append((startX, startY, endX, endY))

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True
        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

    info = [
        ("Exit", totalUp),
        ("Enter", totalDown),
        ("Status", status),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, trackableObjects, totalUp, totalDown


st.sidebar.title("Video Mode [Webcam : Video Upload]")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Webcam", "Video"])

if app_mode == "Webcam":
    st.header("Webcam Input")
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    # process_video(cap)

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    cap.release()
    cv2.destroyAllWindows()

    # st.title("Webcam Live Feed")
    # run_webcam = st.button("Start Webcam")
    # if run_webcam:
    #     vs = VideoStream(src=0).start()
    #     time.sleep(2.0)
    #     ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    #     trackableObjects = {}
    #     totalUp = 0
    #     totalDown = 0
    #     fps = FPS().start()

    #     net = cv2.dnn.readNetFromCaffe('mobilenet_ssd/MobileNetSSD_deploy.prototxt',
    #                                    'mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

    #     while True:
    #         frame = vs.read()
    #         frame, trackableObjects, totalUp, totalDown = process_frame(frame, net, ct, trackableObjects, totalUp, totalDown)

    #         st.image(frame, channels="BGR")

    #         if st.button("Stop Webcam"):
    #             break

    #     vs.stop()
    #     fps.stop()
    #     cv2.destroyAllWindows()

else:
    st.markdown("<h1 style='text-align: center; color: yellow;'>ENHANCED PASSENGER COUNTING IN AIRCRAFTS: LEVERAGING PEOPLE TRACKING AND ANALYSIS WITH <span style='color: yellow;'>OPENCV</span></h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image('.\image\pesawat3.jpg',
                 caption="Pesawat", use_column_width=True)

    with col2:
        st.image('.\image\pramugari.jpg', caption="Perhitungan penumpang",
                 use_column_width=True)

    st.markdown("")
    st.markdown("<div style='text-align: center;'>Aplikasi web ini akan membantu anda berapa orang pengunjung tempat anda secara praktis, hanya dengan mengandalkan kamera, kami bisa secara terus menerus menghitung berapa orang yang masuk dan berapa orang yang keluar.<div>", unsafe_allow_html=True)
    st.markdown('')

    uploaded_file = st.file_uploader(
        "Masukkan video (video harus direkam dari posisi atas kepala, video harus ada orang yang lewat, disarankan ada banyak orang yang lewat.)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackableObjects = {}
        totalUp = 0
        totalDown = 0

        net = cv2.dnn.readNetFromCaffe('mobilenet_ssd/MobileNetSSD_deploy.prototxt',
                                       'mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

        vs = cv2.VideoCapture(video_path)
        fps = FPS().start()

        frame_placeholder = st.empty()

        while True:
            ret, frame = vs.read()
            if not ret:
                break

            frame, trackableObjects, totalUp, totalDown = process_frame(
                frame, net, ct, trackableObjects, totalUp, totalDown)

            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0)  # Adjust the speed of the video display

        vs.release()
        fps.stop()
        cv2.destroyAllWindows()
