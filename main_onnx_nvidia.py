import onnxruntime as ort
import numpy as np
import cupy as cp
import pyautogui
import pygetwindow
import gc
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import (cv2, non_max_suppression, xyxy2xywh)
import dxcam
import torch
import configparser


config = configparser.ConfigParser()
config.read('config.cfg')
config = config['DEFAULT']

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.left = 0
        self.right = width
        self.top = 0
        self.bottom = height


def main():
    aimKey = config.get('aimkey', '0')
    aimKey = [int(key) for key in aimKey.split(',')]

    screenWidth = config.getint('screenWidth', 320)
    screenHeight = config.getint('screenHeight', 320)

    # Portion of screen to be captured (This forms a square/rectangle around the center of screen)    
    fovWidth = config.getint('fovWidth', 320)
    fovHeight = config.getint('fovHeight', 320)

    # For use in games that are 3rd person and character model interferes with the autoaim
    # EXAMPLE: Fortnite and New World
    aaRightShift = config.getint('aaRightShift', 0)

    # Autoaim mouse movement amplifier
    sensX = config.getfloat('sensX', 0.8)
    sensY = config.getfloat('sensY', 0.8)

    # Person Class Confidence
    confidence = config.getfloat('confidence', 0.6)

    # What key to press to quit and shutdown the autoaim
    quitKey = (config.getint('quitKey', 0))

    # If you want to main slightly upwards towards the head
    headshot_mode = config.getboolean('headshot_mode', False)
    headshot_offset = config.getfloat('headshot_offset', 0.38)

    # Displays the Corrections per second in the terminal
    cpsDisplay = config.getboolean('cpsDisplay', False)

    # Set to True if you want to get the visuals
    visuals = config.getboolean('visuals', True)

    target_fps = config.getint('target_fps', 60)


    videoGameWindow = Rectangle(screenWidth, screenHeight)

    # Setting up the screen shots
    sctArea = {"mon": 1, "top": videoGameWindow.top + (videoGameWindow.height - fovHeight) // 2,
                         "left": aaRightShift + ((videoGameWindow.left + videoGameWindow.right) // 2) - (fovWidth // 2),
                         "width": fovWidth,
                         "height": fovHeight}

    # Starting screenshoting engine
    left = aaRightShift + \
        ((videoGameWindow.left + videoGameWindow.right) // 2) - (fovWidth // 2)
    top = videoGameWindow.top + \
        (videoGameWindow.height - fovHeight) // 2
    right, bottom = left + 320, top + 320

    region = (left, top, right, bottom)

    camera = dxcam.create(device_idx=0, region=region, max_buffer_len=5120, output_color="BGR")
    if camera is None:
        print("""DXCamera failed to initialize. Some common causes are:
        1. You are on a laptop with both an integrated GPU and discrete GPU. Go into Windows Graphic Settings, select python.exe and set it to Power Saving Mode.
         If that doesn't work, then read this: https://github.com/SerpentAI/D3DShot/wiki/Installation-Note:-Laptops
        2. The game is an exclusive full screen game. Set it to windowed mode.""")
        return
    camera.start(target_fps=target_fps, video_mode=True)

    # Calculating the center Autoaim box
    cWidth = sctArea["width"] / 2
    cHeight = sctArea["height"] / 2

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession('yolov5s320.onnx', sess_options=so, providers=[
                                    'CUDAExecutionProvider'])

    # Main loop Quit if Q is pressed
    last_mid_coord = None
    while win32api.GetAsyncKeyState(quitKey) == 0:

        # Getting Frame
        npImg = np.array(camera.get_latest_frame())

        # Normalizing Data
        im = torch.from_numpy(npImg).to('cuda')
        im = torch.movedim(im, 2, 0)
        im = im.half()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        outputs = ort_sess.run(None, {'images': cp.asnumpy(im)})

        im = torch.from_numpy(outputs[0]).to('cpu')

        pred = non_max_suppression(
            im, confidence, confidence, 0, False, max_det=10)

        targets = []
        for i, det in enumerate(pred):
            s = ""
            gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {int(c)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                        1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

        targets = pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

        # If there are people in the center bounding box
        if len(targets) > 0:
            # Get the last persons mid coordinate if it exists
            if last_mid_coord:
                targets['last_mid_x'] = last_mid_coord[0]
                targets['last_mid_y'] = last_mid_coord[1]
                # Take distance between current person mid coordinate and last person mid coordinate
                targets['dist'] = np.linalg.norm(
                    targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                targets.sort_values(by="dist", ascending=True)

            # Take the first person that shows up in the dataframe (Recall that we sort based on Euclidean distance)
            min_diff = float('inf')
            idx = 0
            for i in range(len(targets)):
                diff = abs(targets.iloc[i].current_mid_x - cWidth)
                if diff < min_diff:
                    min_diff = diff
                    idx = i

            xMid = targets.iloc[idx].current_mid_x + aaRightShift
            yMid = targets.iloc[idx].current_mid_y

            box_height = targets.iloc[idx].height
            if headshot_mode:
                offset = box_height * headshot_offset
            else:
                offset = box_height * 0.2

            dx = xMid - cWidth
            dy = (yMid - offset) - cHeight
            mouseMove = [dx, dy]

            # Moving the mouse
            if any(win32api.GetAsyncKeyState(key) for key in aimKey):
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(
                    mouseMove[0] * sensX), int(mouseMove[1] * sensY), 0, 0)
            last_mid_coord = [xMid, yMid]

        else:
            last_mid_coord = None

        # See what the bot sees
        if visuals:
            # Loops over every item identified and draws a bounding box
            for i in range(0, len(targets)):
                halfW = round(targets["width"][i] / 2)
                halfH = round(targets["height"][i] / 2)
                midX = targets['current_mid_x'][i]
                midY = targets['current_mid_y'][i]
                (startX, startY, endX, endY) = int(midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)

                # draw the bounding box and label on the frame
                label = "{} {}: {:.2f}%".format(
                    "Human", i,targets["confidence"][i] * 100)
                color = (0,0,255)
                if(idx==i):
                    color = (0,255,0)
                cv2.rectangle(npImg, (startX, startY), (endX, endY),
                              color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(npImg, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                targetCenter = (int((startX+endX)/2), int((startY+endY+11)/2-offset))
                center = (int(cWidth), int(cHeight+11))                
                cv2.line(npImg, center, targetCenter, color, 1)

        # Forced garbage cleanup every second
        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay:
                print("CPS: {}".format(count))
            count = 0
            sTime = time.time()

            # Uncomment if you keep running into memory issues
            # gc.collect(generation=0)

        # See visually what the Aimbot sees
        if visuals:
            cv2.imshow('Live Feed', npImg)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                exit()
    camera.stop()


main()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Please read the below message and think about how it could be solved before posting it on discord.")
        traceback.print_exception(e)
        print(str(e))
        print("Please read the above message and think about how it could be solved before posting it on discord.")
