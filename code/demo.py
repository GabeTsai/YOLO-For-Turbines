import streamlit as st
import torch
import pandas as pd
from PIL import Image
import numpy as np
import os
import gdown
import sys

from utils import non_max_suppression, cells_to_boxes, plot_original
from model import YOLOv3
import config

# Load the YOLOv3 model
@st.cache_resource
def load_model(weights_path, gdrive_url):
    if not os.path.exists(weights_path):
        gdown.download(gdrive_url, weights_path, quiet=False)
    model = torch.load(weights_path)
    model.eval()  # Set model to evaluation mode
    return model

def load_turbine_model(weights_path):
    model = YOLOv3(num_classes = 2, activation = "mish")
    checkpoint = torch.load(weights_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict(model, image):
    anchors = config.ANCHORS
    
    scaled_anchors = torch.tensor(anchors) * (
                    torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )

    image_transforms = config.set_only_image_transforms()
    resize = image_transforms(image = image)
    resized_img = resize['image'].unsqueeze(0)

    with torch.no_grad():
        out = model(resized_img)
        bboxes = [[] for _ in range(out[0].shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_boxes(
                out[i], anchor, grid_size=S, is_pred=True
            )
            print(len(boxes_scale_i))
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    
    nms_boxes = non_max_suppression(
            bboxes[0], iou_threshold=0.45, obj_threshold= config.CONF_THRESHOLD, box_format="center",
        )
    
    obj_preds = []
    for box in nms_boxes:
        label = config.COCO_LABELS[int(box[5])]  # Make sure the class label index is an integer
        obj_preds.append((label, box[4]))
    
    obj_preds_df = pd.DataFrame(obj_preds, columns=["Class", "Confidence"])

    plotted_image = plot_original(image, np.array(resized_img[0]).transpose((1, 2, 0)), nms_boxes, class_list = config.COCO_LABELS)
    
    return plotted_image, obj_preds_df

# Streamlit App Interface
st.title("YOLOv3 Object Detection on MSCOCO Dataset")

# Upload image or select example image
uploaded_image = st.file_uploader("Choose an image from the Common Objects in Context (COCO) dataset...", type=["jpg", "jpeg", "png"])

# Example images
example_images_folder = f"{config.PROJ_FOLDER}/streamlit_examples"
example_images = [f for f in os.listdir(example_images_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
selected_example_image = st.selectbox("Or select an example image:", ["None"] + example_images)

gdrive_url = "https://drive.google.com/file/d/1bY3gEaYC0-i_s-sQdzf7Ix_DqZqlYmQ6/view?usp=sharing"
weights_path = f"{config.MODEL_FOLDER}/YOLOv3COCO.pth"  # Replace with your model's path
model = load_model(weights_path, gdrive_url)  # Load the model

# Load and process the selected image
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = np.array(image)
elif selected_example_image != "None":
    image_path = os.path.join(example_images_folder, selected_example_image)
    image = Image.open(image_path).convert('RGB')
    st.image(image, caption = selected_example_image, use_column_width=True)
    image = np.array(image)
else:
    image = None

# Run YOLOv3 inference if an image is provided
if image is not None:
    result_image, scores_df = predict(model, image)

    if len(scores_df) == 0:
        st.markdown("<p style='text-align: center; font-size: 15px;'>No objects detected.</p>", unsafe_allow_html=True)
    else:
        st.image(result_image, caption="Detected Objects", use_column_width=True)
        html = scores_df.to_html(index=False)
        centered_html = f"""
        <div style="display: flex; justify-content: center;">
            {html}
        </div>
        """
        st.markdown(centered_html, unsafe_allow_html=True)
    