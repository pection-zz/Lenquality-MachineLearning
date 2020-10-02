
from pyueye_example_camera import Camera
from pyueye_example_utils import FrameThread
from pyueye_example_gui import PyuEyeQtApp, PyuEyeQtView
from PyQt4 import QtGui

from pyueye import ueye

import cv2
import numpy as np

def process_image(self, image_data):

    image = image_data.as_1d_image()    
    return QtGui.QImage(image.data,
                        image_data.mem_info.width,
                        image_data.mem_info.height,
                        QtGui.QImage.Format_RGB888)

def main():

    app = PyuEyeQtApp()

    # a basic qt window
    view = PyuEyeQtView()
    view.show()
    view.user_callback = process_image

    # camera class to simplify uEye API access
    cam = Camera()
    cam.init()
    cam.set_colormode(ueye.IS_CM_RGB8_PACKED)
    cam.set_Saturation(1,1)
    cam.set_aoi(0,0, 1280, 1024)
    cam.alloc()
    cam.capture_video()

    thread = FrameThread(cam, view)
    thread.start()

    app.exit_connect(thread.stop)
    app.exec_()

    thread.stop()
    thread.join()

    cam.stop_video()
    cam.exit()

if __name__ == "__main__":
    main()

