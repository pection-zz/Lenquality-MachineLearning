from pyueye import ueye
import numpy as np
import cv2
 
def main():
    # init camera
	hcam = ueye.HIDS(0)
	ret = ueye.is_InitCamera(hcam, None)
	print(ret)

	# set color mode
	ret = ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)
	print(ret)

	# set region of interest
	width = 1280
	height = 1080
	rect_aoi = ueye.IS_RECT()
	rect_aoi.s32X = ueye.int(0)
	rect_aoi.s32Y = ueye.int(0)
	rect_aoi.s32Width = ueye.int(width)
	rect_aoi.s32Height = ueye.int(height)
	ueye.is_AOI(hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

	# allocate memory
	mem_ptr = ueye.c_mem_p()
	mem_id = ueye.int()
	bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
	ret = ueye.is_AllocImageMem(hcam, width, height, bitspixel,
	                        mem_ptr, mem_id)

	# set active memory region
	ret = ueye.is_SetImageMem(hcam, mem_ptr, mem_id)
	
	# continuous capture to memory
	ret = ueye.is_CaptureVideo(hcam, ueye.IS_DONT_WAIT)

	# get data from camera and display
	lineinc = width * int((bitspixel + 7) / 8)
	while True:
		img = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
		img = np.reshape(img, (height, width, 3))
		cv2.imshow('uEye Python Example (q to exit)', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
	cv2.destroyAllWindows()

	# cleanup
	ret = ueye.is_StopLiveVideo(hcam, ueye.IS_FORCE_VIDEO_STOP)

	ret = ueye.is_ExitCamera(hcam)


if __name__ == '__main__':
    main()
