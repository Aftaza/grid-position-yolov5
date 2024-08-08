import cv2
import time
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# used to record the time when we processed last frame 
prev_frame_time = 0

# used to record the time at which we processed current frame 
new_frame_time = 0

# font which we will be using to display FPS 
font = cv2.FONT_HERSHEY_SIMPLEX

center = {'x': 0, 'y': 0}
gridbox = {
    'x': [],
    'y': [],
}

def draw_grid(img, grid_shape, color=(0, 255, 255), thickness=2):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # Clear previous grid lines
    gridbox['x'].clear()
    gridbox['y'].clear()

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        gridbox['x'].append(x)
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        gridbox['y'].append(y)
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def identify_grid_position(center, gridbox):
    x_pos = sum(center['x'] > x for x in gridbox['x'])
    y_pos = sum(center['y'] > y for y in gridbox['y'])
    return (x_pos, y_pos)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    new_frame_time = time.time()
    
    for (x, y, w, h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        # draw center box
        center = {'x': int(x + w/2), 'y': int(y + h/2)}
        cv2.circle(img, (center['x'], center['y']), 5, (0, 0, 255), -1)  
    
    fps = 1 / (new_frame_time - prev_frame_time) 
    prev_frame_time = new_frame_time
    
    fps = int(fps) 
    fps = f"{fps} fps"
    
    img = draw_grid(img, (3, 3))
    grid_position = identify_grid_position(center, gridbox)
    
    cv2.putText(img, fps, (7, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Grid Position: {grid_position}", (7, 60), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
    
    # Display an image in a window 
    cv2.imshow('img', img) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Close the window 
cap.release() 
cv2.destroyAllWindows()
