import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

#Function Definitions
def mean_iou_single(y_true, y_pred):
  keras_meaniou = tf.keras.metrics.MeanIoU(num_classes=2)
  y_true = np.ravel(np.round(y_true))
  y_pred = np.ravel(np.round(y_pred))
  keras_meaniou.update_state(y_true, y_pred)
  return keras_meaniou.result().numpy()

#App Start
def main():
  st.title('Medical Image Segmentation')
  readme_text = st.markdown("Welcome! This application demonstrates the use of the deep learning models U-Net and \
    U-Net++ for the task of segmentation of nuclei in cell images. This was created as a part of my case study on \
    medical image segmentation for which you can find a detailed blog here: https://medium.com/@kriz17/medical-image-segmentation-3093bef449a5")
  readme_text2 = st.markdown("Click on 'Run the app' to see the app working.")
  st.sidebar.title("What to do?")
  app_mode = st.sidebar.selectbox("Choose the app mode",
      ["Show instructions", "Run the app"])
  if app_mode == "Show instructions":
      st.sidebar.success('To continue select "Run the app".')
  elif app_mode == "Run the app":
      readme_text.empty()
      readme_text2.empty()
      run_the_app()

def run_the_app():

  def get_prediction(img):
    input_shape = input_details[0]['shape']
    input_data = img.reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    pred_mask = interpreter.get_tensor(output_details[0]['index'])[0]
    return pred_mask

  df_images, df_masks = load_df()
  interpreter, input_details, output_details = get_interpreter()

  st.subheader('Captured Image to be Segmented')
  index = st.sidebar.slider('Select the image you want to segment', 0, len(df_images))
  st.image(df_images[index], width=400)
  flag = st.sidebar.button('Get Predicted Mask')

  if flag:
    st.subheader('Segmented Masks (by U-Net)')
    start = time.time()
    pred_mask = get_prediction(df_images[index])
    end = time.time()
    time_taken = round(end-start, 3)
    mask = df_masks[index]
    st.success('(Time taken for prediction by the Model = {} sec)'.format(time_taken))
    col1, col2 = st.columns(2)
    with col1:
      st.write('Human Annotated Mask (Ground Truth)')
      st.image(mask, width=320)
    with col2:
      st.write('Predicted Mask by the Model')
      st.image(np.round(pred_mask), width=320)

    score = mean_iou_single(mask, pred_mask)
    score = np.round(score, 3)
    display = 'Mean IoU Score for the prediction: ' + str(score)
    st.success(display)

@st.cache
def load_df():
  with open('df_images.pkl', 'rb') as f : df_images = pickle.load(f)
  with open('df_masks.pkl', 'rb') as f : df_masks = pickle.load(f)
  df_images = df_images.astype('float32')
  df_masks = df_masks.astype('float32')
  return df_images, df_masks

@st.cache
def get_interpreter():
  interpreter = tf.lite.Interpreter(model_path="tflite_unet_1_quant.tflite")
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  return interpreter, input_details, output_details

main()
