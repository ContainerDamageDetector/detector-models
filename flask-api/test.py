import flask
from flask import request
import pandas as pd
import boto3
from urllib.parse import urlparse
from matplotlib import style
style.use('seaborn')
import xgboost as xgb
import numpy as np

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

def predictRecoverPrice(image_name, bulged_dice=None, cut_dice=None, dented_dice=None, hole_dice=None, rust_dice=None,
                        bulged=None, cut=None, dented=None, hole=None, rust=None, minor=None, moderate=None,
                        severe=None, left_side=None, right_side=None, front_side=None, rear_side=None, top_side=None,
                        bottom_side=None, corner_post=None):
    image_np_damage_type = load_image_into_numpy_array(image_name)
    output_dict = run_inference_for_single_image(model_damage_type, image_np_damage_type)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_damage_type,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_damage_type,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    # display(Image.fromarray(image_np))
    im_width, im_height = Image.fromarray(image_np_damage_type).size
    # print('height- ',im_height)
    # print('width- ',im_width)
    # This is the way I'm getting my coordinates
    boxes = output_dict['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        #
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name_damage_type = category_index_damage_type[output_dict['detection_classes'][i]]['name']
            cordinates_damage_type = boxes[i].tolist()
            # print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
            print("The damage type of the image (", image_name, ") :", class_name_damage_type, " - ",
                  round(scores[i] * 100), "%")
            print("class_name_damage_type", class_name_damage_type)
            # print(cordinates_damage_type)


    image_np_severity = load_image_into_numpy_array(image_name)
    output_dict = run_inference_for_single_image(model_severity_type, image_np_severity)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_severity,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_severity_type,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    # display(Image.fromarray(image_np))
    im_width, im_height = Image.fromarray(image_np_severity).size
    # print('height- ',im_height)
    # print('width- ',im_width)
    # This is the way I'm getting my coordinates
    boxes = output_dict['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        #
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name_severity = category_index_severity_type[output_dict['detection_classes'][i]]['name']
            # print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
            cordinates_severity = boxes[i].tolist()
            print("The severity of the image (", image_name, ") :", class_name_severity, " - ", round(scores[i] * 100),
                  "%")
            print("class_name_severity", class_name_severity)
            # print(cordinates_severity)



    image_np_container_sides = load_image_into_numpy_array(image_name)
    output_dict = run_inference_for_single_image(model_container_sides, image_np_container_sides)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_container_sides,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index_container_sides,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    # display(Image.fromarray(image_np))
    im_width, im_height = Image.fromarray(image_np_container_sides).size
    # print('height- ',im_height)
    # print('width- ',im_width)
    # This is the way I'm getting my coordinates
    boxes = output_dict['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = output_dict['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh = .5
    # iterate over all objects found
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        #
        if scores is None or scores[i] > min_score_thresh:
            # boxes[i] is the box which will be drawn
            class_name_container_sides = category_index_container_sides[output_dict['detection_classes'][i]]['name']
            cordinates_container_sides = boxes[i].tolist()
            # print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
            print("The container sides of the image (", image_name, ") :", class_name_container_sides, " - ",
                  round(scores[i] * 100), "%")
            print("class_name_container_sides", class_name_container_sides)
            # print(cordinates_container_sides)

    print("class_name_damage_type :: ", class_name_damage_type)
    print("class_name_severity :: ", class_name_severity)
    print("class_name_container_sides :: ", class_name_container_sides)

    y_pred = cordinates_damage_type

    y_true = cordinates_container_sides


    # Dice similarity function
    def dice(pred, true, k=1):
        intersection = np.sum(pred[true == k]) * 2.0
        dice = intersection / (np.sum(pred) + np.sum(true))
        return dice


    dice_score = dice(y_pred, y_true, k=255)  # 255 in my case, can be 1

    print("Dice Coefficient: {}".format(dice_score))

    if (class_name_damage_type == 'bulged'):
        bulged_dice = dice_score
        cut_dice = 0.0
        dented_dice = 0.0
        hole_dice = 0.0
        rust_dice = 0.0
        bulged = 1
        cut = 0
        dented = 0
        hole = 0
        rust = 0
        unknown_damage_type = 0
        print("bulged_dice ", bulged_dice)

    elif (class_name_damage_type == 'cut'):
        cut_dice = dice_score
        bulged_dice = 0.0
        dented_dice = 0.0
        hole_dice = 0.0
        rust_dice = 0.0
        cut = 1
        bulged = 0
        dented = 0
        hole = 0
        rust = 0
        unknown_damage_type = 0
        print("cut_dice ", cut_dice)

    elif (class_name_damage_type == 'dented'):
        dented_dice = dice_score
        bulged_dice = 0.0
        cut_dice = 0.0
        hole_dice = 0.0
        rust_dice = 0.0
        dented = 1
        bulged = 0
        cut = 0
        hole = 0
        rust = 0
        unknown_damage_type = 0
        print("dented_dice", dented_dice)

    elif (class_name_damage_type == 'hole'):
        hole_dice = dice_score
        bulged_dice = 0.0
        cut_dice = 0.0
        dented_dice = 0.0
        rust_dice = 0.0
        hole = 1
        bulged = 0
        cut = 0
        dented = 0
        rust = 0
        unknown_damage_type = 0
        print("hole_dice", hole_dice)

    elif (class_name_damage_type == 'rust'):
        rust_dice = dice_score
        bulged_dice = 0.0
        cut_dice = 0.0
        dented_dice = 0.0
        hole_dice = 0.0
        rust = 1
        bulged = 0
        cut = 0
        dented = 0
        hole = 0
        unknown_damage_type = 0
        print("rust_dice ", rust_dice)

    elif (class_name_damage_type == ''):
        bulged_dice = 0.0
        cut_dice = 0.0
        dented_dice = 0.0
        hole_dice = 0.0
        rust_dice = 0.0
        unknown_damage_type = 1

    else:
        print("*******")

    if (class_name_severity == 'minor'):
        minor = 1
        moderate = 0
        severe = 0
        unknown_severity = 0
        print(class_name_severity, minor, moderate, severe, unknown_severity)

    elif (class_name_severity == 'moderate'):
        moderate = 1
        minor = 0
        severe = 0
        unknown_severity = 0
        print(class_name_severity, minor, moderate, severe, unknown_severity)

    elif (class_name_severity == 'severe'):
        severe = 1
        minor = 0
        moderate = 0
        unknown_severity = 0
        print(class_name_severity, minor, moderate, severe, unknown_severity)

    elif (class_name_severity == ''):
        minor = 0
        moderate = 0
        severe = 0
        unknown_severity = 1
        print(class_name_severity, minor, moderate, severe, unknown_severity)

    if (class_name_container_sides == 'left_side'):
        left_side = 1
        right_side = 0
        front_side = 0
        rear_side = 0
        top_side = 0
        bottom_side = 0
        corner_post = 0
        unknown_container_side = 0
        print("left side", left_side)

    elif (class_name_container_sides == 'right_side'):
        right_side = 1
        left_side = 0
        front_side = 0
        rear_side = 0
        top_side = 0
        bottom_side = 0
        corner_post = 0
        unknown_container_side = 0

    elif (class_name_container_sides == 'front_side'):
        front_side = 1
        right_side = 0
        left_side = 0
        rear_side = 0
        top_side = 0
        bottom_side = 0
        corner_post = 0
        unknown_container_side = 0

    elif (class_name_container_sides == 'rear_side'):
        rear_side = 1
        front_side = 0
        right_side = 0
        left_side = 0
        top_side = 0
        bottom_side = 0
        corner_post = 0
        unknown_container_side = 0

    elif (class_name_container_sides == 'top_side'):
        top_side = 1
        rear_side = 0
        front_side = 0
        right_side = 0
        left_side = 0
        bottom_side = 0
        corner_post = 0
        unknown_container_side = 0

    elif (class_name_container_sides == 'bottom_side'):
        bottom_side = 0
        top_side = 0
        rear_side = 0
        front_side = 0
        right_side = 0
        left_side = 0
        corner_post = 0
        unknown_container_side = 0

    elif (class_name_container_sides == 'corner_post'):
        corner_post = 1
        bottom_side = 0
        top_side = 0
        rear_side = 0
        front_side = 0
        right_side = 0
        left_side = 0
        unknown_container_side = 0

    elif (class_name_container_sides == ''):
        unknown_container_side = 1
        left_side = 0
        right_side = 0
        front_side = 0
        rear_side = 0
        top_side = 0
        bottom_side = 0
        corner_post = 0

        print('bulged_dice', bulged_dice)
        print('cut_dice', cut_dice)
        print('dented_dice', dented_dice)
        print('hole_dice', hole_dice)
        print('rust_dice', rust_dice)
        print('bulged', bulged)
        print('cut', cut)
        print('dented', dented)
        print('hole', hole)
        print('rust', rust)
        print('unknown_damage_type', unknown_damage_type)
        print('minor', minor)
        print('moderate', moderate)
        print('severe', severe)
        print('unknown_severity', unknown_severity)
        print('left_side', left_side)
        print('right_side', right_side)
        print('front_side', front_side)
        print('rear_side', rear_side)
        print('top_side', top_side)
        print('bottom_side', bottom_side)
        print('corner_post', corner_post)
        print('unknown_container_side', unknown_container_side)

    model = xgb.Booster()
    model.load_model('recover_price/saved_model.model')

    data = {'bulged_dice': bulged_dice, 'cut_dice': cut_dice, 'dented_dice': dented_dice, 'hole_dice': hole_dice,
            'rust_dice': rust_dice, 'bulged': bulged, 'cut': cut, 'dented': dented, 'hole': hole, 'rust': rust,
            'unknown_damage_type': unknown_damage_type, 'minor': minor, 'moderate': moderate, 'severe': severe,
            'unknown_severity': unknown_severity, 'left_side': left_side, 'right_side': right_side,
            'front_side': front_side, 'rear_side': rear_side, 'top_side': top_side, 'bottom_side': bottom_side,
            'corner_post': corner_post, 'unknown_container_side': unknown_container_side}
    index = [0]
    new_df = pd.DataFrame(data, index)
    new_data_matrix = xgb.DMatrix(data=new_df)

    new_pred = model.predict(new_data_matrix)
    print("The container price : ", new_pred)

    return {
        "new_pred": {new_pred}
    }

app = flask.Flask(__name__)


# app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    return "<h1>DETECTOR MODEL</h1>"

@app.route('/estimateRecoverPrice', methods=['POST'])
def estimate_recover_price():
    s3 = boto3.resource('s3')

    url = request.form.get('url')
    parsed_url = urlparse(url)
    bucket_name = 'container-damage-detector'
    key = parsed_url.path.lstrip('/')
    print(key)

    local_filename = (f'images/recover_price/{key.split("/")[-1]}')

    print(local_filename)
    s3.Bucket(bucket_name).download_file(key, local_filename)

    data = predictRecoverPrice(local_filename)
    # data_list = data.tolist()
    # json_data = json.dumps(data_list)
    # response = {"data": data}

    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response

app.run(port=6000)