"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""

import os
import random
import numpy as np
import PIL
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import playground
from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main():
    data_path = 'dataset.npz'
    classes_path = 'classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    data = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'

    image_data, boxes = data['images'], data['boxes']
    
    boxes = process_boxes(image_data, boxes)

    anchors = YOLO_ANCHORS

    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    model_body, model = create_model(anchors, class_names)
    """
    print('Starting training!')
    train_gen(
        model,
        class_names,
        anchors,
        image_data,
        boxes,
        detectors_mask,
        matching_true_boxes
    )
    """
    from datetime import datetime
    start = datetime.now()
    draw_random(model_body,
        class_names,
        anchors,
        generator(image_data,boxes,detectors_mask,matching_true_boxes, batch_size=1))
    end = datetime.now()
    elapsed = end - start
    print('Finished with {} seconds elapsed!'.format(elapsed.total_seconds()))

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_boxes(images, boxes):
    orig_size = np.array([images.shape[1], images.shape[2]])
    orig_size = np.expand_dims(orig_size, axis=0)
    
    # Box preprocessing.
    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
    boxes = [box.reshape((-1, 5)) for box in boxes]
    # Get extents as y_min, x_min, y_max, x_max, class for comparision with
    # model output.
    boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

    # Get box parameters as x_center, y_center, box_width, box_height, class.
    boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
    boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
    boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
    boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
    boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

    # find the max number of boxes
    max_boxes = 0
    for boxz in boxes:
        if boxz.shape[0] > max_boxes:
            max_boxes = boxz.shape[0]

    # add zero pad for training
    for i, boxz in enumerate(boxes):
        if boxz.shape[0]  < max_boxes:
            zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
            boxes[i] = np.vstack((boxz, zero_padding))

    return np.array(boxes)

def process_images(images):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]
    return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model
    
def train_val_split(images, boxes, masks, matches, val_split=0.1):
    train_len = round(images.shape[0] * (1.-val_split))
    val_len = images.shape[0] - train_len
    
    train_images = np.empty((train_len,)+images.shape[1:], dtype=images.dtype)
    train_boxes = np.empty((train_len,)+boxes.shape[1:], dtype=boxes.dtype)
    train_masks = np.empty((train_len,)+masks.shape[1:], dtype=masks.dtype)
    train_matches = np.empty((train_len,)+matches.shape[1:], dtype=matches.dtype)
    
    val_images = np.empty((val_len,)+images.shape[1:], dtype=images.dtype)
    val_boxes = np.empty((val_len,)+boxes.shape[1:], dtype=boxes.dtype)
    val_masks = np.empty((val_len,)+masks.shape[1:], dtype=masks.dtype)
    val_matches = np.empty((val_len,)+matches.shape[1:], dtype=matches.dtype)
    
    indices = [i for i in range(0, images.shape[0])]
    random.shuffle(indices)
    for i,index in enumerate(indices[:train_len]):
        train_images[i] = images[index]
        train_boxes[i] = boxes[index]
        train_masks[i] = masks[index]
        train_matches[i] = matches[index]
    for i,index in enumerate(indices[train_len:]):
        val_images[i] = images[index]
        val_boxes[i] = boxes[index]
        val_masks[i] = masks[index]
        val_matches[i] = matches[index]
    train_data = (train_images, train_boxes, train_masks, train_matches)
    val_data = (val_images, val_boxes, val_masks, val_matches)
    return train_data, val_data

def generator(image_data, boxes, detectors_mask, matching_true_boxes, batch_size=32):
    im_shape = (batch_size,) + image_data.shape[1:]
    box_shape = (batch_size,) + boxes.shape[1:]
    mask_shape = (batch_size,) + detectors_mask.shape[1:]
    match_shape = (batch_size,) + matching_true_boxes.shape[1:]
    while True:
        image_batch = np.empty(im_shape, dtype=image_data.dtype)
        box_batch = np.empty(box_shape, dtype=boxes.dtype)
        mask_batch = np.empty(mask_shape, dtype=detectors_mask.dtype)
        match_batch = np.empty(match_shape, dtype=matching_true_boxes.dtype)
        for i in range(batch_size):
            index = random.randint(0,len(image_data)-1)
            image_batch[i] = image_data[index]
            box_batch[i] = boxes[index]
            mask_batch[i] = detectors_mask[index]
            match_batch[i] = matching_true_boxes[index]
        processed_images = process_images(image_batch)
        yield [processed_images, box_batch, mask_batch, match_batch], np.zeros((batch_size,1))
        
def train_gen(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, val_split=0.1):
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.
    train_data, val_data = train_val_split(image_data, 
                                           boxes, 
                                           detectors_mask, 
                                           matching_true_boxes, 
                                           val_split=val_split)
    train_len = round(len(image_data)*(1.-val_split))
    val_len = len(image_data) - train_len
    print('Created train-val split!')
    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
    model.fit_generator(generator=generator(*train_data, batch_size=32),
                        steps_per_epoch=train_len//32,
                        validation_data = generator(*val_data, batch_size=32),
                        validation_steps = val_len//32,
                        epochs=5,
                        callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit_generator(generator=generator(*train_data, batch_size=8),
                        steps_per_epoch=train_len//8,
                        validation_data = generator(*val_data, batch_size=8),
                        validation_steps = val_len // 8,
                        epochs=30,
                        callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    model.fit_generator(generator=generator(*train_data, batch_size=8),
                        steps_per_epoch=train_len//8,
                        validation_data = generator(*val_data, batch_size=8),
                        validation_steps = val_len // 8,
                        epochs=30,
                        callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('trained_stage_3.h5')

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5,
              callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('trained_stage_3.h5')
    
def draw_random(model_body, class_names, anchors, data_gen):
    draw(model_body, class_names, anchors, data_gen.__next__()[0][0])

def draw(model_body, class_names, anchors, image_data, weights_name='trained_stage_3_best.h5'):
    '''
    Draw bounding boxes on image data
    '''
    image_data = np.array([np.expand_dims(image, axis=0)
        for image in image_data])
    
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.3, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # To display (pauses the program):
        plt.imshow(image_with_boxes, interpolation='nearest')
        plt.show()



if __name__ == '__main__':
    _main()
