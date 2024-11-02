import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize deques to handle color points of different colors for each hand
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
bpoints_2 = [deque(maxlen=1024)]
gpoints_2 = [deque(maxlen=1024)]
rpoints_2 = [deque(maxlen=1024)]
ypoints_2 = [deque(maxlen=1024)]

# Indexes for tracking points in color arrays for each hand
blue_index, green_index, red_index, yellow_index = 0, 0, 0, 0
blue_index_2, green_index_2, red_index_2, yellow_index_2 = 0, 0, 0, 0

# Dilation kernel
kernel = np.ones((5, 5), np.uint8)

# Colors and color index for each hand
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex_hand1 = -1  # -1 means no color selected
colorIndex_hand2 = -1

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
buttons = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW"]
for i, (label, color) in enumerate(zip(buttons, [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (0,255,255)])):
    cv2.rectangle(paintWindow, (40 + i*120, 1), (140 + i*120, 65), color, 2)
    cv2.putText(paintWindow, label, (55 + i*120, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Paint', 800, 600)

# Mediapipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
ret = True

def ensure_deque_exists(points_list, index):
    """Ensure that the deque exists at the specified index"""
    while len(points_list) <= index:
        points_list.append(deque(maxlen=512))
    return points_list[index]

def clear_hand_drawing(hand_index):
    """Clear drawings for specific hand"""
    global blue_index, green_index, red_index, yellow_index
    global blue_index_2, green_index_2, red_index_2, yellow_index_2
    global colorIndex_hand1, colorIndex_hand2
    global bpoints, gpoints, rpoints, ypoints
    global bpoints_2, gpoints_2, rpoints_2, ypoints_2
    
    if hand_index == 0:
        # Clear first hand's drawings only
        for deq in bpoints:
            deq.clear()
        for deq in gpoints:
            deq.clear()
        for deq in rpoints:
            deq.clear()
        for deq in ypoints:
            deq.clear()
        bpoints = [deque(maxlen=1024)]
        gpoints = [deque(maxlen=1024)]
        rpoints = [deque(maxlen=1024)]
        ypoints = [deque(maxlen=1024)]
        blue_index = green_index = red_index = yellow_index = 0
        colorIndex_hand1 = -1
    else:
        # Clear second hand's drawings only
        for deq in bpoints_2:
            deq.clear()
        for deq in gpoints_2:
            deq.clear()
        for deq in rpoints_2:
            deq.clear()
        for deq in ypoints_2:
            deq.clear()
        bpoints_2 = [deque(maxlen=1024)]
        gpoints_2 = [deque(maxlen=1024)]
        rpoints_2 = [deque(maxlen=1024)]
        ypoints_2 = [deque(maxlen=1024)]
        blue_index_2 = green_index_2 = red_index_2 = yellow_index_2 = 0
        colorIndex_hand2 = -1

def redraw_canvas():
    """Redraw the entire canvas with current points"""
    # Redraw the white background and buttons
    paintWindow[:] = 255
    for i, (label, color) in enumerate(zip(buttons, [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (0,255,255)])):
        cv2.rectangle(paintWindow, (40 + i*120, 1), (140 + i*120, 65), color, 2)
        cv2.putText(paintWindow, label, (55 + i*120, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw lines from first hand
    points_lists = [(bpoints, colors[0]), (gpoints, colors[1]), 
                   (rpoints, colors[2]), (ypoints, colors[3])]
    for points, color in points_lists:
        for pt_list in points:
            for k in range(1, len(pt_list)):
                if pt_list[k - 1] is None or pt_list[k] is None:
                    continue
                cv2.line(paintWindow, pt_list[k - 1], pt_list[k], color, 2)
    
    # Draw lines from second hand
    points_lists_2 = [(bpoints_2, colors[0]), (gpoints_2, colors[1]), 
                    (rpoints_2, colors[2]), (ypoints_2, colors[3])]
    for points, color in points_lists_2:
        for pt_list in points:
            for k in range(1, len(pt_list)):
                if pt_list[k - 1] is None or pt_list[k] is None:
                    continue
                cv2.line(paintWindow, pt_list[k - 1], pt_list[k], color, 2)

def get_color_index(x_coord):
    """Get color index based on x coordinate"""
    if 160 <= x_coord <= 255:
        return 0  # Blue
    elif 275 <= x_coord <= 370:
        return 1  # Green
    elif 390 <= x_coord <= 485:
        return 2  # Red
    elif 505 <= x_coord <= 600:
        return 3  # Yellow
    return -1

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw color and clear buttons
    for i, color in enumerate([(0,0,0), (255,0,0), (0,255,0), (0,0,255), (0,255,255)]):
        cv2.rectangle(frame, (40 + i*120, 1), (140 + i*120, 65), color, 2)
    
    for i, label in enumerate(buttons):
        cv2.putText(frame, label, (55 + i*120, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Hand landmark detection
    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = [(int(lm.x * 640), int(lm.y * 480)) for lm in hand_landmarks.landmark]
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            fore_finger, thumb = landmarks[8], landmarks[4]
            center = fore_finger
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            # Select color or clear with either hand
            if center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear button
                    clear_hand_drawing(idx)  # Clear only the current hand's drawings
                    redraw_canvas()
                else:
                    new_color_index = get_color_index(center[0])
                    if new_color_index != -1:
                        if idx == 0 and new_color_index != colorIndex_hand2:
                            colorIndex_hand1 = new_color_index
                        elif idx == 1 and new_color_index != colorIndex_hand1:
                            colorIndex_hand2 = new_color_index

            # Draw points based on selected color for each hand
            elif thumb[1] - center[1] < 30:
                current_color = colorIndex_hand1 if idx == 0 else colorIndex_hand2
                if current_color == -1:
                    continue

                if idx == 0:
                    if current_color == 0:
                        bpoints.append(deque(maxlen=512))
                        blue_index += 1
                    elif current_color == 1:
                        gpoints.append(deque(maxlen=512))
                        green_index += 1
                    elif current_color == 2:
                        rpoints.append(deque(maxlen=512))
                        red_index += 1
                    elif current_color == 3:
                        ypoints.append(deque(maxlen=512))
                        yellow_index += 1
                else:
                    if current_color == 0:
                        bpoints_2.append(deque(maxlen=512))
                        blue_index_2 += 1
                    elif current_color == 1:
                        gpoints_2.append(deque(maxlen=512))
                        green_index_2 += 1
                    elif current_color == 2:
                        rpoints_2.append(deque(maxlen=512))
                        red_index_2 += 1
                    elif current_color == 3:
                        ypoints_2.append(deque(maxlen=512))
                        yellow_index_2 += 1
            else:
                current_color = colorIndex_hand1 if idx == 0 else colorIndex_hand2
                if current_color == -1:
                    continue

                if idx == 0:
                    if current_color == 0:
                        ensure_deque_exists(bpoints, blue_index).appendleft(center)
                    elif current_color == 1:
                        ensure_deque_exists(gpoints, green_index).appendleft(center)
                    elif current_color == 2:
                        ensure_deque_exists(rpoints, red_index).appendleft(center)
                    elif current_color == 3:
                        ensure_deque_exists(ypoints, yellow_index).appendleft(center)
                else:
                    if current_color == 0:
                        ensure_deque_exists(bpoints_2, blue_index_2).appendleft(center)
                    elif current_color == 1:
                        ensure_deque_exists(gpoints_2, green_index_2).appendleft(center)
                    elif current_color == 2:
                        ensure_deque_exists(rpoints_2, red_index_2).appendleft(center)
                    elif current_color == 3:
                        ensure_deque_exists(ypoints_2, yellow_index_2).appendleft(center)

    # Draw lines on frame
    # First hand's drawings
    points_colors = [
        (bpoints, colors[0]), (gpoints, colors[1]), 
        (rpoints, colors[2]), (ypoints, colors[3])
    ]
    for points, color in points_colors:
        for pt_list in points:
            for k in range(1, len(pt_list)):
                if pt_list[k - 1] is None or pt_list[k] is None:
                    continue
                cv2.line(frame, pt_list[k - 1], pt_list[k], color, 2)

    # Second hand's drawings
    points_colors_2 = [
        (bpoints_2, colors[0]), (gpoints_2, colors[1]), 
        (rpoints_2, colors[2]), (ypoints_2, colors[3])
    ]
    for points, color in points_colors_2:
        for pt_list in points:
            for k in range(1, len(pt_list)):
                if pt_list[k - 1] is None or pt_list[k] is None:
                    continue
                cv2.line(frame, pt_list[k - 1], pt_list[k], color, 2)

    # Update paint window
    redraw_canvas()

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()