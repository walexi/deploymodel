from fastai import *
from fastai.vision import *
from moviepy.editor import ImageSequenceClip
!pip install moviepy
# ! pip install pytube
# from pytube import YouTube

# from ipywidgets import Video

# !pip3 install opencv-python #For python 3.x
# !conda install opencv-python #If you directly install in anaconda with all dependencies.

import cv2
import cv2.cv2 as cv
import numpy as np
from os.path import isfile, join



def getFrame(vidcap):
  hasFrame, image = vidcap.read();
  if hasFrame: return hasFrame, image
  return hasFrame, None

def video2Frame(input_path, sec=0):
  if input_path is not None and os.path.isfile(input_path) and input_path.lower().endswith('.mp4'):
    vidcap = cv2.VideoCapture(input_path);
    frameRate = vidcap.get(cv.CAP_PROP_FPS)
#     GetCaptureProperty(vidcap, cv2.CV_CAP_PROP_FPS)
    frameArray = []
    hasFrame, image = getFrame(vidcap)
    frameArray.append(image)
    while hasFrame:
      hasFrame, image = getFrame(vidcap)
#       print(hasFrame)
      if image is not None: frameArray.append(image)
  else:
    raise Exception('Pls input a valid video file');
  
  return frameArray, frameRate;



  def export2Video(frames: list, output: str, preds: list, classes: list, fps=25, filename='output.avi'):
  '''
  Method to reconstruct video from frames and predictions from model
  
  Parameters
  
  --------------------
  
  frames   : list
             an array of frames/images in the right order or sorting
  output   : str
             output path to save exported video
  preds    : list
             an array of probabilities returned by running model inference, in the correct order/sorting
  classes  : list
             an array of classes in the right order
  fps      : double
             frames per second, default is 0.5
  filename : str
             file name of exported video, default is output.avi
  '''
  if isinstance(frames,list) and isinstance(preds, list) and isinstance(classes, list) and os.path.exists(output) and len(frames)>0:
#     print()
    size = (frames[0].shape[1], frames[0].shape[0]);
    finalframes = []
    for i,img  in enumerate(frames):
      offset=0
      if img is not None:
        for j,pred in enumerate(preds[i]):
          offset += int(size[0] / len(classes)) - 80
          img = cv2.putText(img, classes[j] + ':'+str(round(pred,3)), (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)          
#           print(img.shape)
          finalframes.append(img)
    out = ImageSequenceClip(finalframes, fps=fps)
    out.write_videofile("output.mp4") 
      
      
    # extract frames from test video
myarr, fps = video2Frame('/content/video.mp4')


# run inference on each of the extract frames and append predictions to an array
myresults = []
for img in myarr:
  if img is not None:
    ## convert image to tensor
    img = pil2tensor(img, dtype=np.float32).div(255)
    ## reorder tensor size, bring channel to first dim
#     print(img.shape)
    img = img.permute(0,1,2)
#     print(img.shape)
    ##create fastai image
    img = Image(img)
    myresults.append(learn.predict(img)[2].tolist())


export2Video(myarr, '/content', myresults,learn.data.classes, fps=25)