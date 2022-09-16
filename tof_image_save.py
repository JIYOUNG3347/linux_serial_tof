import cv2
import numpy as np
import serial
import ctypes
from ctypes import cdll
import os
import time

crclib = cdll.LoadLibrary('./crc.dll')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cmd_get_dis_amp = [0xF5, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE9, 0xDF, 0xE8, 0x9E]
LIDARPORT = 'COM4'  # 장치관리자 확인 후 설정
BAUDRATE = 10000000  # 바꾸지 마!//

ser = serial.Serial(
    port=LIDARPORT,
    baudrate=BAUDRATE,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1)  # 0 (non-block) / 0.1(100ms)


def checkPck(response_data, calc_response_crc):
    bit4 = (calc_response_crc >> 24) & 0xff
    bit3 = (calc_response_crc >> 16) & 0xff
    bit2 = (calc_response_crc >> 8) & 0xff
    bit1 = calc_response_crc & 0xff
    cmp = (bit4 == response_data[-1]) & (bit3 == response_data[-2]) & (bit2 == response_data[-3]) & (
                bit1 == response_data[-4])
    return cmp


if ser.isOpen():
    print("Serial port is opened")
    running = True
else:
    running = False

sample_num = 0
captured_num = 0

while running:

    ser.write(bytearray(cmd_get_dis_amp))
    rxdata = ser.read()

    if rxdata == b'\xfa':   # 0xFa ---> bytes(0xfa) ---> b'\xfa'
        start = time.time()

        response_data = [0xfa]
        response_data.extend(ser.read(1 + 2 + 38480 + 4))  # 시작 바이트 읽고, type (1), length (2), data (38480), crc (4)

        crc_data = (ctypes.c_uint8 * len(response_data[:-4]))(*response_data[:-4])  # 타입 변환 (불분명한 type에서 c++ uint8로)
        crc_data = crclib.calcCrc32_32(crc_data, ctypes.c_uint32(len(response_data[:-4])))

        crc_result = ctypes.c_uint32(crc_data).value
        cmp = checkPck(response_data, crc_result)

        if not cmp:
            continue



        dis_amp_arr = np.array(response_data[1 + 1 + 2 + 80:-4])  # 시작 바이트 (1), type (1), length (2), header data (80), ~, crc (4)
        l = len(dis_amp_arr)
        try:
            dis_arr = ((dis_amp_arr[1::4] & 0xff) << 8) | (dis_amp_arr[0::4] & 0xff)
            amp_arr = ((dis_amp_arr[3::4] & 0xff) << 8) | (dis_amp_arr[2::4] & 0xff)  # dis[하위|상위] amp[하위|상위] 요렇게 옴. 즉 (상위 바이트 << 8 + 하위 바이트)를 해야 resolution 값이 뽑힘
        except:
            continue



        dis_arr = np.array(dis_arr, dtype=float)
        amp_arr = np.array(amp_arr, dtype=float)

        dis_arr = (np.where(dis_arr > 8000, 8000, dis_arr) / 8000) * 255
        amp_arr = (np.where(amp_arr > 2000, 2000, amp_arr) / 2000) * 255

        dis_img = np.reshape(dis_arr, (60, 160))
        amp_img = np.reshape(amp_arr, (60, 160))

        dis_img = cv2.resize(dis_img, (256, 256), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        amp_img = cv2.resize(amp_img, (256, 256), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        dis_img = dis_img.astype('uint8')
        amp_img = amp_img.astype('uint8')
        zer_img = np.zeros((256,256,1), np.uint8)
        frame = cv2.merge((zer_img, dis_img, amp_img))
        frame2 = cv2.merge((zer_img, dis_img))


        # read frame from webcam
        sample_num = 1

        cv2.imshow('img', frame)

        if sample_num == 1:
            captured_num = captured_num + 1
            cv2.imwrite('Dataset/img/img' + str(captured_num) + '.jpg', frame)
            cv2.imwrite('Dataset/amp/amp' + str(captured_num) + '.jpg', amp_img)
            cv2.imwrite('Dataset/depth/dis' + str(captured_num) + '.jpg', dis_img)


        if cv2.waitKey(1) == ord('q'):
            break