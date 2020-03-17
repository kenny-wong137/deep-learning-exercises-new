import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xml
import os
import sys
import PIL

class Box:
    
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    
    @property
    def width(self):
        return max(self.xmax - self.xmin, 0)
    
    @property
    def height(self):
        return max(self.ymax - self.ymin, 0)
    
    @property
    def xcentre(self):
        return 0.5 * (self.xmin + self.xmax)
    
    @property
    def ycentre(self):
        return 0.5 * (self.ymin + self.ymax)
    
    @property
    def area(self):
        return self.width * self.height


# Image size is (160, 160). We superimpose a 5x5 grid of cells, where each cell has size (32, 32).
# On each grid cell we place anchors of sizes (32, 32), (64, 64), (96, 96) and (128, 128).

def make_anchor_boxes():
    anchor_boxes = []
    for y_id in range(5):  # first dimension in image array is y, not x
        for x_id in range(5):
            for anchor_size in [32, 64, 96, 128]:
                xmin = (x_id + 0.5) * 32 - 0.5 * anchor_size
                xmax = (x_id + 0.5) * 32 + 0.5 * anchor_size
                ymin = (y_id + 0.5) * 32 - 0.5 * anchor_size
                ymax = (y_id + 0.5) * 32 + 0.5 * anchor_size
                anchor_boxes.append(Box(xmin, xmax, ymin, ymax))
    return anchor_boxes


def calc_iou(box1, box2):
    intersection_box = Box(xmin=max(box1.xmin, box2.xmin), xmax=min(box1.xmax, box2.xmax),
                           ymin=max(box1.ymin, box2.ymin), ymax=min(box1.ymax, box2.ymax))
    intersection_area = intersection_box.area
    union_area = box1.area + box2.area - intersection_area
    return intersection_area / union_area


def calc_xywh(real_box, anchor_box):
    xcentre_offset = (real_box.xcentre - anchor_box.xcentre) / 32
    ycentre_offset = (real_box.ycentre - anchor_box.ycentre) / 32
    log_width_ratio = np.log(real_box.width / anchor_box.width)
    log_height_ratio = np.log(real_box.height / anchor_box.height)
    return (xcentre_offset, ycentre_offset, log_width_ratio, log_height_ratio)


def perturb_with_xywh(anchor_box, xcentre_offset, ycentre_offset, log_width_ratio, log_height_ratio):
    new_xcentre = anchor_box.xcentre + xcentre_offset * 32
    new_ycentre = anchor_box.ycentre + ycentre_offset * 32
    new_width = anchor_box.width * np.exp(log_width_ratio)
    new_height = anchor_box.height * np.exp(log_height_ratio)
    
    new_xmin = new_xcentre - 0.5 * new_width
    new_xmax = new_xcentre + 0.5 * new_width
    new_ymin = new_ycentre - 0.5 * new_height
    new_ymax = new_ycentre + 0.5 * new_height
    return Box(new_xmin, new_xmax, new_ymin, new_ymax)


def compute_average_of_boxes(boxes, weights):
    ave_xmin = np.clip(np.average([box.xmin for box in boxes], weights=weights), 0, 160)
    ave_xmax = np.clip(np.average([box.xmax for box in boxes], weights=weights), 0, 160)
    ave_ymin = np.clip(np.average([box.ymin for box in boxes], weights=weights), 0, 160)
    ave_ymax = np.clip(np.average([box.ymax for box in boxes], weights=weights), 0, 160)
    return Box(ave_xmin, ave_xmax, ave_ymin, ave_ymax)


def get_training_targets(real_box, anchor_boxes):
    iou_values = []  # IoU's of *anchor* with real box
    xywh_values = [] # The centre and width/height offsets of real box relative to anchor
    
    for anchor_box in anchor_boxes:
        iou_values.append(calc_iou(real_box, anchor_box))
        xywh_values.append(calc_xywh(real_box, anchor_box))
    
    iou_values = np.resize(np.array(iou_values), (5, 5, 4, 1))
    xywh_values = np.resize(np.array(xywh_values), (5, 5, 4, 4))
    return iou_values, xywh_values


def get_predicted_box(iou_pred, xywh_pred, anchor_boxes):
    iou_pred = np.resize(np.array(iou_pred), new_shape=(100,))
    weights = np.square(iou_pred)   # weight predicted boxes by (predicted IoU)^2.
    
    xywh_pred = np.resize(np.array(xywh_pred), new_shape=(100, 4))
    boxes = []
    for box_id in range(100):
        xcentre_offset, ycentre_offset, log_width_ratio, log_height_ratio = xywh_pred[box_id]
        box = perturb_with_xywh(anchor_boxes[box_id], xcentre_offset, ycentre_offset,
                                log_width_ratio, log_height_ratio)
        boxes.append(box)
    
    return compute_average_of_boxes(boxes, weights)


def load_image(filestem, images_path, annotations_path):            
    raw_image = np.array(PIL.Image.open('{}/{}.jpg'.format(images_path, filestem)))
    orig_height, orig_width = raw_image.shape[0], raw_image.shape[1]
    
    if len(raw_image.shape) == 2:   # black and white
        raw_image = np.stack([raw_image for _ in range(3)], axis=-1)
    
    image = tf.image.resize(raw_image, size=(160, 160))
    image = image / 127.5 - 1.0

    annotation = xml.etree.ElementTree.parse('{}/{}.xml'.format(annotations_path, filestem))
    xmin = int(annotation.findtext('./object/bndbox/xmin')) * 160 / orig_width
    xmax = int(annotation.findtext('./object/bndbox/xmax')) * 160 / orig_width
    ymin = int(annotation.findtext('./object/bndbox/ymin')) * 160 / orig_height
    ymax = int(annotation.findtext('./object/bndbox/ymax')) * 160 / orig_height
    box = Box(xmin, xmax, ymin, ymax)
    
    return image, box


def random_crop(image, box):
    left_crop = np.random.randint(low=0, high=(min(16, box.xmin) + 1))
    right_crop = np.random.randint(low=0, high=(min(16, 160 - box.xmax) + 1))
    top_crop = np.random.randint(low=0, high=(min(16, box.ymin) + 1))
    bottom_crop = np.random.randint(low=0, high=(min(16, 160 - box.ymax) + 1))
    
    cropped_image = image[top_crop:(160 - bottom_crop), left_crop:(160 - right_crop), :]
    cropped_image = tf.image.resize(cropped_image, size=(160, 160))
    
    new_xmin = (box.xmin - left_crop) / (160 - left_crop - right_crop) * 160
    new_xmax = (box.xmax - left_crop) / (160 - left_crop - right_crop) * 160
    new_ymin = (box.ymin - top_crop) / (160 - top_crop - bottom_crop) * 160
    new_ymax = (box.ymax - top_crop) / (160 - top_crop - bottom_crop) * 160
    new_box = Box(new_xmin, new_xmax, new_ymin, new_ymax)
    
    return cropped_image, new_box


def random_flip(image, box):
    if np.random.uniform() < 0.5:
        flipped_image = image[:, ::-1, :]
        new_box = Box((160 - box.xmax), (160 - box.xmin), box.ymin, box.ymax)
        return flipped_image, new_box
    else:
        return image, box
    

def make_dataset(filestems, images_path, annotations_path, anchor_boxes, jitter, batch_size):
    
    def generator():
        while True:
            filestem = np.random.choice(filestems)
            image, box = load_image(filestem, images_path, annotations_path)
            
            if jitter:
                image, box = random_crop(image, box)
                image, box = random_flip(image, box)
            
            iou_values, xywh_values = get_training_targets(box, anchor_boxes)
            
            yield (image, iou_values, xywh_values)
        
    return tf.data.Dataset.from_generator(
                    generator,
                    output_types=(tf.float32, tf.float32, tf.float32),
                    output_shapes=((160, 160, 3), (5, 5, 4, 1), (5, 5, 4, 4))) \
                .batch(batch_size, drop_remainder=True) \
                .repeat(None)


def create_train_test_data(images_path, annotations_path, anchor_boxes,
                           num_test_images, train_batch_size, test_batch_size):
    filestems = [file.split('.')[0] for file in os.listdir(annotations_path)]
    np.random.shuffle(filestems)
    train_filestems = filestems[:-num_test_images]
    test_filestems = filestems[-num_test_images:]
    
    train_data = make_dataset(train_filestems, images_path, annotations_path, anchor_boxes,
                              jitter=True, batch_size=train_batch_size)
    test_data = make_dataset(test_filestems, images_path, annotations_path, anchor_boxes,
                             jitter=False, batch_size=test_batch_size)
    
    return train_data, test_data


def build_model():
    inputs = tf.keras.Input(shape=(160, 160, 3), dtype=tf.float32)
    
    # 1. Feature extraction base (pretrained).
    pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    
    for layer in pretrained_model.layers:
        layer.trainable = False

    feature_extraction_layers = tf.keras.models.Model(
            pretrained_model.input, pretrained_model.get_layer('block_16_project_BN').output)
    
    feature_values = feature_extraction_layers(inputs)
    
    # 2. Predictors of IoU and xywh offsets for each anchor.
    iou_hidden = tf.keras.layers.Conv2D(filters=128, kernel_size=3,
                                        padding='same', activation='relu')(feature_values)
    iou = tf.keras.layers.Conv2D(filters=4, kernel_size=1, activation='sigmoid')(iou_hidden)
    iou = tf.keras.layers.Reshape((5, 5, 4, 1))(iou)
    
    xywh_hidden = tf.keras.layers.Conv2D(filters=256, kernel_size=3,
                                         padding='same', activation='relu')(feature_values)
    xywh = tf.keras.layers.Conv2D(filters=16, kernel_size=1, activation=None)(xywh_hidden)
    xywh = tf.keras.layers.Reshape((5, 5, 4, 4))(xywh)
    
    return tf.keras.models.Model(inputs, [iou, xywh])


@tf.function
def run_train_step(model, optimizer, images, iou_true, xywh_true, xywh_loss_multiplier):
    with tf.GradientTape() as tape:
        iou_pred, xywh_pred = model(images, training=True)
        iou_loss = tf.keras.losses.BinaryCrossentropy()(iou_true, iou_pred)
        xywh_loss = tf.keras.losses.MeanSquaredError()(xywh_true, xywh_pred,
                                                       sample_weight=tf.squeeze(iou_true))
            # only care about getting xywh correct when anchor box overlaps well with real box
        loss = iou_loss + xywh_loss_multiplier * xywh_loss
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def demo_predictions(model, images, iou_true, xywh_true, anchor_boxes, savepath):
    iou_pred, xywh_pred = model(images, training=False)
    
    for idx in range(images.shape[0]):
        image = images[idx]
        image = 0.5 * np.array(image) + 0.5
        plt.imshow(image)
        
        box_pred = get_predicted_box(iou_pred[idx], xywh_pred[idx], anchor_boxes)
        rect = plt.Rectangle((box_pred.xmin, box_pred.ymin),
                             width=box_pred.width, height=box_pred.height,
                             fill=False, edgecolor='r', linewidth=2.5)
        plt.gca().add_patch(rect)
        
        box_true = get_predicted_box(iou_true[idx], xywh_true[idx], anchor_boxes) # any idx works
        rect = plt.Rectangle((box_true.xmin, box_true.ymin),
                             width=box_true.width, height=box_true.height,
                             fill=False, edgecolor='b', linewidth=2.5)
        plt.gca().add_patch(rect)

        plt.savefig('{}/prediction{:02}.png'.format(savepath, idx))
        plt.show()


def main(num_train_batches, train_batch_size, test_batch_size, num_test_images,
         learning_rate, xywh_loss_multiplier,
         images_path, annotations_path, predictions_path):
    '''
    :param num_train_batches: number of training batches
    :param train_batch_size: number of images in a training batch
    :param test_batch_size: number of images to predict on in each demo
    :param num_test_images: number of images to reserve for demo predictions
    :param learning_rate: learning rate for Adam optimizer
    :param xywh_loss_multiplier: weighting to give to loss for centre and width/height offsets
                            relative to loss for intersection-over-unions
    :param images_path: folder in which images are saved
    :param annotations_path: folder in which xml bounding box annotations are saved
    :param predictions_path: folder to save prediction images to 
    '''
    anchor_boxes = make_anchor_boxes()
    train_data, test_data = create_train_test_data(
            images_path, annotations_path, anchor_boxes,
            num_test_images, train_batch_size, test_batch_size)
    test_iter = iter(test_data)

    model = build_model()
    optimizer = tf.optimizers.Adam(lr=learning_rate)
    losses = []
    
    for batch_id, (images, iou_true, xywh_true) in enumerate(train_data):
        loss = run_train_step(model, optimizer, images, iou_true, xywh_true, xywh_loss_multiplier)
        losses.append(loss)
        
        if (batch_id + 1) % 25 == 0:
            print('After {} batches: Loss = {:.3f}'.format(
                    (batch_id + 1), sum(losses) / len(losses)))
            losses = []
            
            images, iou_true, xywh_true = next(test_iter)
            demo_predictions(model, images, iou_true, xywh_true,
                             anchor_boxes, savepath=predictions_path)
            
            
if __name__ == '__main__':
    images_path = sys.argv[1]
    annotations_path = sys.argv[2]
    predictions_path = sys.argv[3]
    
    main(num_train_batches=250,
         train_batch_size=64,
         test_batch_size=16,
         num_test_images=128,
         learning_rate=0.001,
         xywh_loss_multiplier=5.0,
         images_path=images_path,
         annotations_path=annotations_path,
         predictions_path=predictions_path)
