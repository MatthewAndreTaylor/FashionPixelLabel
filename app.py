from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageFilter
from skimage.segmentation import felzenszwalb
import torch
from utils import to_tensor_lab, remove_small_artifacts
import numpy as np
from collections import defaultdict
import base64
import io
import os

app = Flask(__name__)

binary_model = torch.load("saved_models/DeepLabv3_2.pt")
binary_model.eval()

multi_model = torch.load("saved_models/DeepLabv3_7.pt")
multi_model.eval()

inverse_color_map = {
    0: np.array((0, 0, 0)),  # background
    1: np.array((128, 0, 0)),  # skin
    2: np.array((0, 128, 0)),  # hair
    3: np.array((128, 128, 0)),  # tshirt
    4: np.array((0, 0, 128)),  # shoes
    5: np.array((128, 0, 128)),  # pants
    6: np.array((0, 128, 128)),  # dress
}

black_white_map = {
    0: np.array((0, 0, 0)),
    1: np.array((255, 255, 255)),
}


def get_superpixel_mask(image, segments) -> list:
    superpixel_mask = np.zeros(image.shape[:2])
    for segment in np.unique(segments):
        if segment <= 1:
            continue
        superpixel_mask[segments == segment] = segment
    return superpixel_mask


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/report", methods=["GET"])
def report():
    return render_template("report.html")


@app.route("/report_md", methods=["GET"])
def report_md():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    return send_file(readme_path)


@app.route("/gaussian", methods=["POST"])
def gaussian():
    file = request.files["file"]
    img = Image.open(file)
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})


@app.route("/binary", methods=["POST"])
def binary():
    file = request.files["file"]
    image = Image.open(file)

    numpy_image = np.array(image)[:, :, :3]
    image_tensor = to_tensor_lab(numpy_image)
    pred_mask = binary_model(image_tensor.cuda())
    pred_mask = pred_mask.argmax(dim=1).cpu().numpy()[0]
    pred_mask = remove_small_artifacts(pred_mask)
    pred_mask_color = np.zeros(
        (pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8
    )

    for class_idx, color in black_white_map.items():
        pred_mask_color[pred_mask == class_idx] = color

    image = Image.fromarray(pred_mask_color)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})


@app.route("/binary_seg", methods=["POST"])
def binary_seg():
    file = request.files["file"]
    image = Image.open(file)

    numpy_image = np.array(image)[:, :, :3]
    image_tensor = to_tensor_lab(numpy_image)
    pred_mask = binary_model(image_tensor.cuda())
    pred_mask = pred_mask.argmax(dim=1).cpu().numpy()[0]
    pred_mask = remove_small_artifacts(pred_mask)
    numpy_image[pred_mask == 0] = [0, 0, 0]
    image = Image.fromarray(numpy_image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})


@app.route("/multilabel", methods=["POST"])
def multilabel():
    file = request.files["file"]
    image = Image.open(file)

    numpy_image = np.array(image)[:, :, :3]
    image_tensor = to_tensor_lab(numpy_image)
    pred_mask = multi_model(image_tensor.cuda())
    pred_mask = pred_mask.argmax(dim=1).cpu().numpy()[0]
    pred_mask = remove_small_artifacts(pred_mask)
    pred_mask_color = np.zeros(
        (pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8
    )

    for class_idx, color in inverse_color_map.items():
        pred_mask_color[pred_mask == class_idx] = color

    image = Image.fromarray(pred_mask_color)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})


@app.route("/multilabel_super", methods=["POST"])
def multilabel_super():
    file = request.files["file"]
    image = Image.open(file)

    numpy_image = np.array(image)[:, :, :3]
    image_tensor = to_tensor_lab(numpy_image)
    pred_mask = multi_model(image_tensor.cuda())
    pred_mask = pred_mask.argmax(dim=1).cpu().numpy()[0]
    pred_mask = remove_small_artifacts(pred_mask)
    segments_fz = felzenszwalb(numpy_image, scale=100, min_size=100, channel_axis=2)
    superpixel_mask = get_superpixel_mask(numpy_image, segments_fz)
    prediction = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)

    for superpixel in np.unique(superpixel_mask):
        freq = defaultdict(int)
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if superpixel_mask[i, j] == superpixel:
                    freq[pred_mask[i, j]] += 1
        try:
            max_key = max(freq, key=freq.get)
            prediction[superpixel_mask == superpixel] = inverse_color_map[max_key]
        except:
            continue

    image = Image.fromarray(prediction)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})


if __name__ == "__main__":
    app.run(debug=True)
