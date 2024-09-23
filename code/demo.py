import streamlit as st
import torch
import pandas as pd
from utils import non_max_suppression, cells_to_boxes

# Load the YOLOv3 model
@st.cache_resource
def load_model(weights_path):
    model = torch.load(weights_path)
    model.eval()  # Set model to evaluation mode
    return model

def predict(model, image, confidence_threshold=0.7):
    scaled_anchors = torch.tensor(anchors) * (
                    torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    )

    image_transforms = config.set_only_image_transforms()
    resize = image_transforms(image = image)
    resized_img = resize['image']

    with torch.no_grad():
        out = model(resized_img)
        bboxes = []
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = scaled_anchors[i]
            boxes_scale_i = cells_to_boxes(
                out[i], anchor, grid_size=S, is_pred=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx].append(box)

    nms_boxes = non_max_suppression(
            bboxes, iou_threshold=0.5, obj_threshold= confidence_threshold, box_format="center",
        )
    
    obj_preds = {}
    for box in nms_boxes:
        label = config.COCO_LABELS[int(box[5])]  # Make sure the class label index is an integer
        confidence_score = box[4]
        obj_preds[label] = confidence_score
    
    obj_preds_df = pd.DataFrame(list(obj_preds.items()), columns=["Class Name", "Confidence Score"])
    obj_preds_df["Confidence Score"] = obj_preds_df["Confidence Score"].round(4)

    plotted_image = plot_original(original_image, image, nms_boxes, class_list = config.COCO_LABELS)
    
    return plotted_image, obj_preds_df

# Streamlit App Interface
st.title("YOLOv3 Object Detection")

# Upload image
uploaded_image = st.file_uploader("Choose an image from the Common Objects in Context (COCO) dataset...", type=["jpg", "jpeg", "png"])

# Load model
weights_path = f"{config.MODEL_FOLDER}/YOLOv3COCO.pth"  # Replace with your model's path
model = load_model(weights_path)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = np.array(image)
    st.write("Running inference...")

    # Run YOLOv3 inference
    result_image, scores_df = predict(model, image)

    st.image(result_image, caption="Detected Objects", use_column_width=True)
    # Draw boxes on the image
    if len(boxes) < 0:
        st.write("No objects detected.")
    else:
        st.write(scores_df)
    
    st.write("Inference complete.")