# Pedestrians Movement in a video using OpenCV

✅ Detects pedestrians in a video using OpenCV and
✅ Animates the results inline in a Jupyter Notebook using Matplotlib.

1. cv2: OpenCV for computer vision tasks.
2. matplotlib.pyplot: For plotting frames.
3. matplotlib.animation: To animate each frame of video.
4. IPython.display.HTML: To display HTML animation inline in Jupyter Notebook.
5. rcParams: To change plot/animation settings (like size limit).

rcParams['animation.embed_limit'] = 100

1. Matplotlib by default limits inline animations to 20 MB.
2. Here, you increase the limit to 100 MB so more frames can be embedded in the notebook.

pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

1. Uses a pre-trained Haar Cascade model for full-body detection.
2. Haar cascades are simple and fast but may miss detections in low light or fast movement.

cap = cv2.VideoCapture('video.mp4')

1. Loads your local video file for processing.
2. Each frame will be read using cap.read() later.

def detect_pedestrians(frame)

1. Converts each frame to grayscale (Haar cascades work on grayscale).
2. Detects full bodies using detectMultiScale.
3. Draws green rectangles around detected people.

def update(_)

1. Reads the next frame from the video.
2. If the video ends (ret=False), stops the animation.
3. Resizes the frame to 320×240 (to reduce memory usage).
4. Detects and draws bounding boxes.
5. Displays the frame using imshow.

ani = animation.FuncAnimation(fig, update, frames=100, interval=50)

1. Runs update() every 50 milliseconds, for 100 frames.
2. This creates an animation by looping through 100 frames of the video.

HTML(ani.to_jshtml())

1. Converts the animation into JavaScript-based HTML.
2. to_jshtml() allows it to play inline in Jupyter Notebook (instead of opening a new window).
