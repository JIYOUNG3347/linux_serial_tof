//
// Created by kaylor on 2022/4/6.
//

#ifndef SERIAL_PORT__SERIAL_PORT_H_
#define SERIAL_PORT__SERIAL_PORT_H_

#include <mutex>
#include <thread>
#include "serial_port_stream.h"
#include "time_stamp.h"
class SerialPort {
 public:

  static const int PARITY_NONE = 0;

  static const int PARITY_ODD = 1;

  static const int PARITY_EVEN = 2;

  static const int OK = 1;

  static const int DEV_NOT_FOUND = -1;

  static const int BAUD_NOT_SUPPORTED = -2;

  static const int DATABITS_NOT_SUPPORTED = -3;

  static const int PARITYMODE_NOT_SUPPORTED = -4;

  static const int STOPBITS_NOT_SUPPORTED = -5;

  static const int CONFIG_FAIL = -6;

  static const int NEW_THREAD_FAIL = -7;

  static const int READ_END = 1;

  static const int READ_TIMEOUT = -1;

  static const int READ_BUFFER_FULL = -2;

 private:

  int fd_;

  SerialPortStream stream_;

  std::mutex mutex_;

  std::thread *pthread_;

  bool running_flag_;

  int RTS_flag_;

  int DTR_flag_;

 public:
  SerialPort();
  ~SerialPort();

  int Open(const char *dev, int baud, int dataBits, int parityMode, int stopBits);

  int Close();

  int Write(const char *data, int len);

  int Available();

  int Peek(unsigned char *buf, int len);

  int Read(unsigned char *buf, int len, int timeout);

  int Read(unsigned char *buf, int maxLen, unsigned char *end, int timeout, int *recvLen);

 private:

  int TransformBaud(int baud);

  int TransformDataBits(int dataBits);
  long long getTimestamp();

  bool EndsWith(unsigned char *str, int strLen, unsigned char *end, int endLen);

  void SetRxMode(void);

  void SetTxMode(void);

  friend void* ReceiveThread(void* arg);
};

#endif //SERIAL_PORT__SERIAL_PORT_H_
