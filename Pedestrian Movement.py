#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import rcParams

# Raise limit to 100MB
rcParams['animation.embed_limit'] = 100

pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
cap = cv2.VideoCapture('video.mp4')

def detect_pedestrians(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

fig, ax = plt.subplots()

def update(_):
    ret, frame = cap.read()
    if not ret:
        ani.event_source.stop()
        return
    frame = cv2.resize(frame, (320, 240))  # Resize to keep animation small
    frame = detect_pedestrians(frame)
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=100, interval=50)  # limit to 100 frames
HTML(ani.to_jshtml())


# In[ ]:





# In[ ]:





# In[ ]:




