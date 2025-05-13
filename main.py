import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

cap = cv2.VideoCapture('popo.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

T = []
Y = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) > 500:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        object_position = y*1.796/1080
        T.append(time_sec)
        Y.append(object_position)

        print(f"Time: {time_sec:.2f} sec, Position: {object_position} m")

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(T)
print(Y)
valueT = max(T)
print(height)

t_data = np.array(T)

y_data = np.array(Y)

def true_free_fall_eq(t):
    return 0.5 * 9.81 * t ** 2 + v0_estimated * t + y0_estimated

def free_fall_eq(t, g, v0, y0):
    return 0.5 * g * t**2 + v0 * t + y0

params, covariance = curve_fit(free_fall_eq, t_data, y_data)

g_estimated, v0_estimated, y0_estimated = params

print(f"g: {g_estimated:.2f} m/s²")
print(f"v0: {v0_estimated:.2f} m/s")
print(f"y0: {y0_estimated:.2f} m")
print(f"t: {valueT:.2f} s")

t_fit = np.linspace(0, valueT, 100) 
y_fit = free_fall_eq(t_fit, *params) 

plt.scatter(t_data, y_data, label='data', color='blue')
plt.plot(t_fit, y_fit, label='fitting curve', color='red')
try:
    x = np.linspace(0, valueT, 100)
    y = true_free_fall_eq(x)
    plt.plot(x, y, label='true curve', color='green')
except:
    print("Error in plotting true curve")
plt.title(f'g : {g_estimated:.2f} m/s², v0 : {v0_estimated:.2f} m/s, y0 : {y0_estimated:.2f} m')
plt.xlabel("t")
plt.ylabel("s")
plt.legend()
plt.grid(True)
plt.show()
