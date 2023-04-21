import serial
import time
import datetime as dt
import numpy as np
import cv2


# function to get Emissivity from MCU
def get_emissivity():
    ser.write(serial.to_bytes([0xA5, 0x55, 0x01, 0xFB]))
    read = ser.read(4)
    return read[2] / 100


# function to get temperatures from MCU (Celsius degrees x 100)
def get_temp_array(d):
    # getting ambient temperature
    T_a = (int(d[1540]) + int(d[1541]) * 256) / 100

    # getting raw array of pixels temperature
    raw_data = d[4:1540]
    T_array = np.frombuffer(raw_data, dtype=np.int16)

    return T_a, T_array


# function to convert temperatures to pixels on image
def td_to_image(f):
    norm = np.uint8(np.abs(f / 100 - Tmin) * 255 / (Tmax - Tmin))
    # norm = np.uint8(np.abs(f / 100 - Tmin) * 255 / (Tmax - Tmin))
    norm.shape = (24, 32)
    return norm


# Image resizing


def img_resize(img):
    img.shape = (24, 32)
    img = cv2.flip(img, 1)
    return img


########################### Main cycle #################################
# Color map range
Tmax = 50
Tmin = 20
display_ratio = 20

print('Configuring Serial port')
ser = serial.Serial('COM18')
ser.baudrate = 460800

# set baud rate 460800bps
# ser.write(serial.to_bytes([0xA5, 0x15, 0x03, 0xBD]))
# ser.baudrate = 460800

# set frequency of module to 1 Hz
# ser.write(serial.to_bytes([0xA5, 0x25, 0x01, 0xCB]))
# set frequency of module to 8 Hz
# ser.write(serial.to_bytes([0xA5, 0x25, 0x04, 0xCE]))
time.sleep(0.1)

# Starting automatic data colection
ser.write(serial.to_bytes([0xA5, 0x35, 0x02, 0xDC]))
t0 = time.time()

mouse_loc = None
poi_loc = []


# 左键添加关键温度点，右键清空
def mouse_callback(event, x, y, flags, param):
    global mouse_loc
    if event == cv2.EVENT_MOUSEMOVE:
        # print("Mouse moved to:", x, y)
        mouse_loc = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        # print("Mouse left clicked:", x, y)
        poi_loc.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # print("Mouse right clicked:", x, y)
        poi_loc.clear()


try:
    while True:
        # waiting for data frame
        data = ser.read(1544)

        # The data is ready, let's handle it!
        Ta, temp_array = get_temp_array(data)
        temp_array = img_resize(temp_array)
        temp_img = td_to_image(temp_array)
        img = temp_img

        # Find the maximum value and its coordinates in the image
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_array)
        # print("Maximum value:", max_val/100)
        # print("Coordinates of maximum value:", max_loc)

        # Image processing

        # 直方图均衡增强对比度
        # img = cv2.equalizeHist(img)
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1, 1))
        img = clahe.apply(img)

        # img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (32 * display_ratio, 24 * display_ratio),
                         interpolation=cv2.INTER_LANCZOS4)

        # https://pyimagesearch.com/2022/10/17/thermal-vision-measuring-your-first-temperature-from-an-image-with-python-and-opencv/

        # https://docs.opencv.org/4.5.4/d3/d50/group__imgproc__colormap.html
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

        text = 'Tmin = {:+.1f} Tmax = {:+.1f} FPS = {:.2f}'.format(temp_array.min() / 100, temp_array.max() / 100,
                                                                   1 / (time.time() - t0 + 1E-9))
        cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)

        # Draw crosshair at max_loc with max_val
        max_loc = (max_loc[0] * display_ratio, max_loc[1] * display_ratio)
        cv2.drawMarker(img, max_loc, (90, 255, 90), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(
            img, f"{max_val / 100:.1f}", (max_loc[0] + 10, max_loc[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (90, 255, 90), 1)

        # Draw crosshair at mouse_loc with mouse_val
        if mouse_loc is not None:
            mouse_val = temp_array[int(
                mouse_loc[1] / display_ratio)][int(mouse_loc[0] / display_ratio)]
            cv2.drawMarker(img, mouse_loc, (90, 255, 90),
                           cv2.MARKER_CROSS, 10, 1)
            cv2.putText(
                img, f"{mouse_val / 100:.1f}", (mouse_loc[0] + 10, mouse_loc[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (90, 255, 90), 1)

        for poi in poi_loc:
            poi_val = temp_array[int(
                poi[1] / display_ratio)][int(poi[0] / display_ratio)]
            cv2.drawMarker(img, poi, (90, 255, 90),
                           cv2.MARKER_CROSS, 10, 1)
            cv2.putText(
                img, f"{poi_val / 100:.1f}", (poi[0] + 10, poi[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 255, 90),
                1)

        cv2.imshow('Output', img)

        # if 's' is pressed - saving of picture
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            fname = 'pic_' + dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.jpg'
            cv2.imwrite(fname, img)
            print('Saving image ', fname)

        # print(f"fps={1 / (time.time() - t0 + 1E-9)}")
        t0 = time.time()

        # Mouse callback
        cv2.setMouseCallback('Output', mouse_callback)

except KeyboardInterrupt:
    # to terminate the cycle
    ser.write(serial.to_bytes([0xA5, 0x35, 0x01, 0xDB]))
    ser.close()
    cv2.destroyAllWindows()
    print(' Stopped')

# just in case
ser.close()
cv2.destroyAllWindows()
