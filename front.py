import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image
import cv2

st.title("YOLO Object Detection with Streamlit")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # 画像の保存
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # YOLOの実行
    # load pretrained model.
    model = YOLO(model="yolov8n.pt")

    # inference
    resPred = model.predict(image_path, save=False, show=True)

    # 結果の表示
    st.write('## YOLO Detection Before')
    st.image(Image.open(image_path), caption='Uploaded Image', use_column_width=True)
 
    st.write('## YOLO Detection Result')

    # Reusltsオブジェクトのplotメソッドでndarrayを取得
    resPlotted = resPred[0].plot()
    # streamlitで画像を表示する場合はRGBで扱っていると思われる。
    # yoloの推論結果(opencv)はBGRで表現されていそう。streamlitで描画するためRGBに変換する
    resPlotted_rgb = cv2.cvtColor(resPlotted, cv2.COLOR_BGR2RGB)

    st.image(resPlotted_rgb)