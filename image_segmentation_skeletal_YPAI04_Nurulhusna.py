#%%
#1. Import the packages
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob, os
from tensorflow import keras
import tensorflow_datasets as tfds

filepath = r"C:\Users\asus\Desktop\ai04_hands_on\exercise2-image-segmentation\carvana-masking-challenge\train"
images = []
masks = []

#%%
#Use os.listdir() method to list down all the image file, then use a for loop to read the images.
"""
for ____ in os.listdir(____):
    function you will use here:
    os.path.join()
    cv2.imread()
    cv2.cvtColor()
    cv2.resize()

Use this for loop, do the same thing for the label. But I suggest you read the label as a grayscale image.
"""
# 2. Load images
image_path = os.path.join(filepath, 'inputs')
for img in os.listdir(image_path):
    # Get the full path of the image file
    full_path = os.path.join(image_path, img)
    # Read the image file based on the full path
    img_np = cv2.imread(full_path)
    # Convert the image from bgr to rgb
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    # Resize the image into 128x128
    img_np = cv2.resize(img_np, (128, 128))
    # Place the image into the empty list
    images.append(img_np)

# 3. Load masks
mask_path = os.path.join(filepath, 'masks_png')
for mask in os.listdir(mask_path):
    # Get the full path of the mask file
    full_path = os.path.join(mask_path, mask)
    # Read the mask file as a grayscale image
    mask_np = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image into 128x128
    mask_np = cv2.resize(mask_np, (128, 128))
    # Place the mask into the empty list
    masks.append(mask_np)

# %%
#4. Convert the list of np array into a full np array
images_np = np.array(images)
masks_np = np.array(masks)
# %%
#5. Data preprocessing
#5.1. Expand the mask dimension to include the channel axis
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#5.2. Convert the mask value into just 0 and 1
converted_masks_np = np.round(masks_np_exp/255)
#5.3. Normalize the images pixel value
normalized_images_np = images_np/255.0
# %%
#6. Perform train test split
from sklearn.model_selection import train_test_split
SEED = 12345
X_train,X_test,y_train,y_test = train_test_split(normalized_images_np,converted_masks_np,shuffle=True,random_state=SEED)

# %%
#7. Convert the numpy array into tensorflow tensors
X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)
# %%
#8. Combine features and labels together to form a zip dataset
train = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
test = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))
# %%
"""
Continue the rest of your exercise here.
"""
#Convert this into prefetch dataset

#9. Define hyperparameters for the tensorflow dataset
TRAIN_LENGTH = len(train)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

#%%
#10. Create a data augmentation layer through subclassing
class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels
  
#11. Build the dataset
train_batches = (
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

# %%
#Build the test dataset
test_batches = test.batch(BATCH_SIZE)

# %%
#Inspect some data
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

#Display some images for inspection
for images, masks in train_batches.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])

# %%
#12. Model development
"""
The plan is to obtain a feature extractor via transfer learning. e.g. we can use MobileNetV2 as the feature extractor.

Then we will proceed to build our own upsampling path using one of the method in tensorflow_examples module.
"""

#12.1 Use thre pretrained as the feature extractor

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

#12.2 Specify the layers as outputs so that they become the correct inputs for our unsampling path
# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#12.3 Instantiate the feature extractor
# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

#%%
#12.4 Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

#%%
#12.5 define a function for the creation of u-net
def unet(output_channels:int):
  #Construct the entire U-Net model using Functional API
  #(A) Input Layer
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  #(B) Down stack (feature extractor)
  # Downsampling through the model
  skips = down_stack(inputs)
  #The last output from feature extractor
  x = skips[-1]
  skips = reversed(skips[:-1])
  #(C) Build the upsampling path
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])
  #(D) Use transpose convolution to perform one last upsampling. This convolution layer will become our output layer as well.
  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128
  outputs = last(x)
  model = keras.Model(inputs=inputs,outputs=outputs)
  return model

#%%
#12.6 Create the U-Net model by calling the function
OUTPUT_CLASSES = 3
model = unet(OUTPUT_CLASSES)
model.summary()
keras.utils.plot_model(model)

# %%
#13. Compile the model
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics='accuracy')

# %%
#14. Create functions to show prediction results
def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

# %%
#15. Create a custom callback function to display results during model training
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# %%
#Creating Tensorboard object
import os, datetime
base_log_path = r"tensorboard_logs\image_segmentation_skeletal"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(log_path)

#%%
#16. Model Training
EPOCHS = 5
VAL_SUBSPLITS = 5

VALIDATION_STEPS = len(test_batches)// BATCH_SIZE // VAL_SUBSPLITS

history = model.fit(train_batches,validation_data=test_batches,validation_steps=VALIDATION_STEPS,epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[DisplayCallback(),tb])

# %%
#17. Model deployment
show_predictions(test_batches,num=3)

#%%
