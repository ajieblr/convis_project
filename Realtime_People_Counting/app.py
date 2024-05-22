import streamlit as st
import cv2
import tempfile
import os
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib.mailer import Mailer
from mylib import config, thread
import imutils
import numpy as np
import datetime
import csv
from imutils.video import FPS
import dlib
from itertools import zip_longest
from moviepy.editor import VideoFileClip, vfx


def process_video(input_path, output_path, prototxt, model, confidence, skip_frames):
    global jumlah_orang
    global info2
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    vs = cv2.VideoCapture(input_path)

    writer = None
    W = None
    H = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty = []
    empty1 = []
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = frame[1]

        if frame is None:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if output_path is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []
        if totalFrames % skip_frames == 0:
            status = "Detecting"
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > confidence:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        # cv2.putText(frame, "-Prediction border - Entrance-", (10, H - 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
                        empty.append(totalUp)
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        empty1.append(totalDown)
                        x.append(len(empty1) - len(empty))

                        if sum(x) >= config.Threshold:
                            cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                Mailer().send(config.MAIL)
                        to.counted = True

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        info = [("Exit", totalUp), ("Enter", totalDown), ("Status", status)]
        info2 = [("Total people inside", len(x))]
        jumlah_orang = info2
        # jumlah_orang.append(info2)

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (265, H - ((i * 20) + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue='')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        if writer is not None:
            writer.write(frame)

        totalFrames += 1
        fps.update()

    fps.stop()
    vs.release()
    if writer is not None:
        writer.release()
    # return info2

def convert_to_mp4(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_videofile(output_file)

# CSS style
centered_title_css = """
<style>
.centered-title {
    text-align: center;
}
.centered-italic {
    text-align: center;
    font-style: italic;
    color: yellow;
}
</style>
"""
# Menyisipkan CSS ke dalam Streamlit
st.markdown(centered_title_css, unsafe_allow_html=True)


def main():
    total_orang = None
    show_video = False

    st.markdown("<h1 style='text-align: center; color: yellow;'>Counting People in a Room: Leveraging People Tracking and Analysis with OpenCV</span></h1>", unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Aplikasi web ini akan membantu anda berapa orang pengunjung tempat anda secara praktis, hanya dengan mengandalkan kamera, kami bisa secara terus menerus menghitung berapa orang yang masuk dan berapa orang yang keluar.<div>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    colom1, colom2 = st.columns(2)
    with colom1:
        st.image('.\image\gadis_pintu-1.jpg', caption="Ilustration Image 1", use_column_width=True)
    with colom2:
        st.image('.\image\gadis_pintu-2.jpg', caption="Ilustration Image 2", use_column_width=True)

    # st.image('.\image\convis.png', caption="Detection Test Image", use_column_width=True)
    # st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('</br>', unsafe_allow_html=True)


    st.sidebar.header("Video Parameters")
    prototxt = st.sidebar.text_input("mobilenet_ssd\MobileNetSSD_deploy.prototxt", "mobilenet_ssd\MobileNetSSD_deploy.prototxt")
    model = st.sidebar.text_input("mobilenet_ssd\MobileNetSSD_deploy.caffemodel", "mobilenet_ssd\MobileNetSSD_deploy.caffemodel")
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.1)
    skip_frames = st.sidebar.slider("Skip Frames", 0, 30, 15)
    max_people = st.sidebar.slider("Jumlah Maksimal Orang", min_value=1, max_value=40, value=10)

    uploaded_file = st.sidebar.file_uploader("Choose a video file")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        output_path = "videos/output.avi"
        # output_path2 = "output/output.mp4"

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Process Video"):
                process_video(input_path, output_path, prototxt, model, confidence, skip_frames)
                st.success("Video processing completed.")
                total_orang = jumlah_orang
        with col2:
            if st.button("Download Processed Video"):
                with open(output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="output.avi",
                        mime="video/x-msvideo"
                    )
        with col3:
            if st.button("Display Processed Video"):
                video_file = "videos/output.avi"
                output_video = "videos/output.mp4"
                # convert_to_mp4(video_file, output_video)
                # with open(video_file, "rb") as file:
                if not video_file.lower().endswith('.mp4'):
                    convert_to_mp4(video_file, output_video)
                else:
                    output_video = video_file

                # st.video(output_video)
                show_video = True
        
        if total_orang is not None:
            st.markdown(
                f"<div style='text-align: center; font-size: 20px; color: yellow;'>Total Orang: {total_orang[0][1]}</div>",
                unsafe_allow_html=True
            )
            if total_orang[0][1] > max_people:
                st.error("Jumlah orang melebihi batas maksimal")
        
        st.markdown(
                f"<h2 style='text-align: center; color: yellow;'>Hasil Video</h2>",
                unsafe_allow_html=True
            )
        if show_video:
            st.video(output_video)

    elif uploaded_file is None:
        st.markdown("<div class='centered-italic'>Mohon upload file video yang ingin anda proses untuk melihat fitur tersembunyi.<div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
