import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t_data = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

y_data = np.array([0, 1, 4.92, 12.5, 19, 30.5, 50])

def true_free_fall_eq(t):
    return 0.5 * 9.81 * t ** 2 + 0 * t + 0

def free_fall_eq(t, g, v0, y0):
    return 0.5 * g * t**2 + v0 * t + y0

params, covariance = curve_fit(free_fall_eq, t_data, y_data)

g_estimated, v0_estimated, y0_estimated = params

print(f"g: {g_estimated:.2f} m/s²")
print(f"v0: {v0_estimated:.2f} m/s")
print(f"y0: {y0_estimated:.2f} m")

t_fit = np.linspace(0, 3, 100) 
y_fit = free_fall_eq(t_fit, *params) 

plt.scatter(t_data, y_data, label='data', color='blue')
plt.plot(t_fit, y_fit, label='fitting curve', color='red')
try:
    x = np.linspace(0, 3, 100)
    y = true_free_fall_eq(x)
    plt.plot(x, y, label='true curve', color='green')
except:
    print("Error in plotting true curve")
plt.title("Freefalling motion")
plt.xlabel("t")
plt.ylabel("s")
plt.legend()
plt.grid(True)
plt.show()
