//
// Created by kaylor on 2022/4/6.
//

#ifndef SERIAL_PORT__SERIAL_PORT_STREAM_H_
#define SERIAL_PORT__SERIAL_PORT_STREAM_H_

class SerialPortStream {
 private:

  unsigned char *buffer_;

  int capacity_;

  int start_;

  int length_;

 public:

  SerialPortStream(int initCapacity = 16);

  ~SerialPortStream();

  int GetLength();

  void Append(char aChar);

  void Append(const char *buf, int len);

  unsigned char Peek();

  int Peek(unsigned char *buf, int len);

  unsigned char Take();

  int Take(unsigned char *buf, int len);

 private:

  void Expand();
};

#endif //SERIAL_PORT__SERIAL_PORT_STREAM_H_
