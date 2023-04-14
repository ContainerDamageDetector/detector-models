import flask
import os
from flask import request
import json

from inferenceutils import *

output_directory = 'inference_graph'
labelmap_path_damage_type = 'damage_type_model/content/labelmap_damage_type.pbtxt'
labelmap_path_severity_type = 'severity_type_model/content/labelmap_severity.pbtxt'
labelmap_path_container_sides = 'container_sides_model/content/labelmap_container_side.pbtxt'


category_index_damage_type = label_map_util.create_category_index_from_labelmap(labelmap_path_damage_type, use_display_name=True)
category_index_severity_type = label_map_util.create_category_index_from_labelmap(labelmap_path_severity_type, use_display_name=True)
category_index_container_sides = label_map_util.create_category_index_from_labelmap(labelmap_path_container_sides, use_display_name=True)
tf.keras.backend.clear_session()

model_damage_type = tf.saved_model.load(f'damage_type_model/content/{output_directory}/saved_model')
model_severity_type = tf.saved_model.load(f'severity_type_model/content/{output_directory}/saved_model')
model_container_sides = tf.saved_model.load(f'container_sides_model/content/{output_directory}/saved_model')


def predictImageData_damage_type(image_name):
    image_np = load_image_into_numpy_array(image_name)
    output_dict = run_inference_for_single_image(model_damage_type, image_np)
    im_width, im_height = Image.fromarray(image_np).size
    print('height- ', im_height)
    print('width- ', im_width)
    # This is the way I'm getting my coordinates
    boxes = output_dict['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    data = []
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name = category_index_damage_type[output_dict['detection_classes'][i]]['name']
            print("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
            data.append({
                "box": boxes[i].tolist(),
                "class": int(output_dict['detection_classes'][i]),
                "class_name": class_name
            })
    return {
        "width": im_width,
        "height": im_height,
        "regions": data
    }

def predictImageData_severity_type(image_name):
    image_np = load_image_into_numpy_array(image_name)
    output_dict = run_inference_for_single_image(model_severity_type, image_np)
    im_width, im_height = Image.fromarray(image_np).size
    print('height- ', im_height)
    print('width- ', im_width)
    # This is the way I'm getting my coordinates
    boxes = output_dict['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    data = []
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name = category_index_severity_type[output_dict['detection_classes'][i]]['name']
            print("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
            data.append({
                "box": boxes[i].tolist(),
                "class": int(output_dict['detection_classes'][i]),
                "class_name": class_name
            })
    return {
        "width": im_width,
        "height": im_height,
        "regions": data
    }

def predictImageData_container_side(image_name):
    image_np = load_image_into_numpy_array(image_name)
    output_dict = run_inference_for_single_image(model_container_sides, image_np)
    im_width, im_height = Image.fromarray(image_np).size
    print('height- ', im_height)
    print('width- ', im_width)
    # This is the way I'm getting my coordinates
    boxes = output_dict['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    data = []
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name = category_index_container_sides[output_dict['detection_classes'][i]]['name']
            print("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
            data.append({
                "box": boxes[i].tolist(),
                "class": int(output_dict['detection_classes'][i]),
                "class_name": class_name
            })
    return {
        "width": im_width,
        "height": im_height,
        "regions": data
    }

app = flask.Flask(__name__)


# app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>DETECTOR MODEL</h1>"


@app.route('/predictDamageType', methods=['POST'])
def predict_damage_type():
    url = request.form.get('url')
    data = predictImageData_damage_type(url)

    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response


@app.route('/predictSevereType', methods=['POST'])
def predict_severity_type():
    url = request.form.get('url')
    data = predictImageData_severity_type(url)

    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response

@app.route('/predictContainerSides', methods=['POST'])
def predict_container_side():
    url = request.form.get('url')
    data = predictImageData_container_side(url)

    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response

app.run()