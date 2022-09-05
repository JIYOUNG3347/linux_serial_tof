#include "string.h"
#include "stdio.h"
#include "iostream"
#include "list"
#include "vector"

// opencv Includes
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "crc_calc.h"
#include "serial_port.h"

using namespace std;
using namespace ComLib;
using namespace cv;
// Command format
const char cmd_get_dis_amp[14] = { 0xF5, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE9, 0xDF, 0xE8, 0x9E };

// Port and Baudrate
string LIDARPORT = "/dev/ttyACM0";			// Check port
int BAUDRATE = 4000000;	// Do not change values

// Global variables
bool running;

// function declaration
bool checkPck(unsigned char* response_data, int calc_response_crc);		// CRC Check function
vector<int> arr(unsigned char* Data, int start_pixcel_1, int start_pixcel_2, int outlier);			// Split data to Distance data and Amplitude data function, outlier handling, Normalize


int main()
{
  // variables
	unsigned char response_data[38488] = {};
  int rxdata;															// Resposne data start byte
  bool cmp;															// CRC Check 

	vector<int> dis_arr;												// Distance data array
	vector<int> amp_arr;												// Amplitude ata array

  SerialPort ser;
  int status = ser.Open("/dev/ttyACM0", BAUDRATE, 8, SerialPort::PARITY_NONE, 1);
  
  if(status == SerialPort::OK){
    cout << "Serial port is opened" << "\r\n";
    running = true;
  }
  else{
    cout << "Serial port is failed" << endl;
    running = false;
  }

  while (running){
    ser.Write(cmd_get_dis_amp, 14);							// Command format
    ser.Read(response_data, 38488, -1);						// Resposnse format -> start byte (1), type (1), length (2), header data (80), ~, crc (4)

    
    std::cout << std::endl << "Data: ";
    for(int i = 0; i < 30000 ; i++){
      int a = response_data[i];
      std::cout << a << " ";
    }
    
    std::cout << std::endl << std::endl;
    

    rxdata = response_data[0];										// start byte

    if (rxdata == 0xfa){
      clock_t start = clock();

      // CRC packet check
			uint32_t crc_result = CrcCalc::calcCrc32_32(response_data, 38484);
      cmp = checkPck(response_data, crc_result);
      cout << cmp << endl;

      if (cmp != true) continue;


			// Split data to Distance data and Amplitude data
			try {
				dis_arr = arr(response_data, 1, 0, 8000);
				amp_arr = arr(response_data, 3, 2, 2000);
			}
			catch (...)
			{
				continue;
			}


			// Mat Data
			Mat dis_mat = (Mat)dis_arr;
			Mat amp_mat = (Mat)amp_arr;
			Mat img;

			// Convert to uint8_t Type
			/*
			If you don't do this first, resize function will throw an error.
			*/
			dis_mat.convertTo(dis_mat, CV_8UC1);
			amp_mat.convertTo(amp_mat, CV_8UC1);


			Mat dis_mat_reshaped = dis_mat.reshape(1, 60);
			Mat amp_mat_reshaped = amp_mat.reshape(1, 60);

			resize(dis_mat_reshaped, dis_mat_reshaped, Size(0, 0), 5, 5, INTER_CUBIC);
			resize(amp_mat_reshaped, amp_mat_reshaped, Size(0, 0), 5, 5, INTER_CUBIC);

			vconcat(dis_mat_reshaped, amp_mat_reshaped, img);


			imshow("img", img);
      clock_t end = clock();
      clock_t result = (end - start) / 1000.0f;
      result /= 1.0f;
      cout << endl << endl << "FPS: " << result << endl;

			if (waitKey(1) == ('q'))
				break;


    }
  }

  
  return  0;
}


// CRC Check function
bool checkPck(unsigned char* response_data, int calc_response_crc)
{
	int bit1, bit2, bit3, bit4;
	bool cmp;

	bit4 = (calc_response_crc >> 24) & 0xff;
	bit3 = (calc_response_crc >> 16) & 0xff;
	bit2 = (calc_response_crc >> 8) & 0xff;
	bit1 = calc_response_crc & 0xff;
	cmp = (bit4 == response_data[38487]) & (bit3 == response_data[38486]) & (bit2 == response_data[38485]) & (bit1 == response_data[38484]);

	return cmp;
}


// Split data to Distance data and Amplitude data function, outlier handling,  Normalize
vector<int> arr(unsigned char* Data, int start_pixcel_1, int start_pixcel_2, int outlier) {
	vector<int> arr;
	int data;
	for (int n = 84; n < 38484; n = n + 4) {

		data = (Data[start_pixcel_1 + n] & 0xff) << 8 | (Data[start_pixcel_2 + n] & 0xff);
		
    // outlier handling
		if (data > outlier) {
			data = outlier;
		}
		
    // Normalize between 0 and 255
		data = (data * 255)/outlier;
		arr.push_back(data);

	}
	return arr;
}
