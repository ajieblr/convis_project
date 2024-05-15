import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'image\demo.png'
DEMO_VIDEO = 'video\demo.mp4'

st.title('Face Mesh App Using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('Facemesh Sidebar')
st.sidebar.subheader('parameter')

app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Run on Image', 'Run on Video'])


# img_file_buffer = st.sidebar.file_uploader('Upload an Image', type=["jpg", "jpeg", "png"])
# if img_file_buffer is not None:
#     image = np.array(Image.open(img_file_buffer))

# else:
#     demo_image = DEMO_IMAGE
#     image = np.array(Image.open(demo_image))

# st.sidebar.text('ori Image')
# st.sidebar.image(image)



if app_mode =='Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Use Webcame')
    record = st.sidebar.checkbox('Record Video')
    if record:
        st.checkbox("Recording", value= True)

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
)
    stframe = st.empty
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('demo.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    with kpi2:
        st.markdown("**Detected People**")
        kpi1_text = st.markdown("0")
    with kpi3:
        st.markdown("**Video Width**")
        kpi1_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)
