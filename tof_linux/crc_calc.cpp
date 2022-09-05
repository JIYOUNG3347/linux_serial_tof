/***
 * Copyright (C) 2017 Espros Photonics Corporation
 *
 * @addtogroup crc
 */
#include "crc_calc.h"

namespace ComLib
{

	///generator polynom
	static const uint32_t polynom = 0x04C11DB7;
	static const uint32_t initValue = 0xFFFFFFFF;
	static const uint32_t xorValue = 0x00000000;

	/**
	 * @brief Calculate CRC32 of a 8 bit buffer
	 *
	 * This function calculates the CRC32 checksum for the given buffer.
	 *
	 * @param data Pointer to the data buffer
	 * @param size Size of the data buffer in Bytes
	 * @return CRC32 checksum
	 */
	uint32_t CrcCalc::calcCrc32(const uint8_t* data, const uint32_t size)
	{
		uint32_t crc = initValue;

		for (uint32_t i = 0; i < size; i++)
		{
			crc = calcCrc32Uint8(crc, data[i]);
		}
		return crc ^ xorValue;
	}

	/**
	 * @brief Calculate CRC32 of a 32 bit buffer
	 *
	 * This function calculates the CRC32 checksum for the given buffer.
	 *
	 * @param data Pointer to the data buffer
	 * @param size Size of the data buffer as count of 32bit values
	 * @return CRC32 checksum
	 */
	uint32_t CrcCalc::calcCrc32(const uint32_t* data, const uint32_t size)
	{
		uint32_t crc = initValue;

		for (uint32_t i = 0; i < size; i++)
		{
			crc = calcCrc32Uint32(crc, data[i]);
		}
		return crc ^ xorValue;
	}


	/**
	 * @brief Calculate CRC32 of a 8 bit buffer
	 *
	 * This function calculates the CRC32 checksum for the given buffer.
	 *
	 * @param data Pointer to the data buffer
	 * @param size Size of the data buffer in Bytes
	 * @return CRC32 checksum
	 */
	uint32_t CrcCalc::calcCrc32_32(const uint8_t* data, const uint32_t size)
	{
		uint32_t crc = initValue;

		for (uint32_t i = 0; i < size; i++)
		{
			crc = calcCrc32Uint32(crc, data[i]);
		}
		return crc ^ xorValue;
	}




	/**
	 * @brief Calculate CRC32 of one Byte
	 *
	 * This function calculates the CRC32 checksum of one Byte. It is only
	 * used internally as private function
	 *
	 * @param crc Actual CRC value
	 * @param byte Byte to use
	 * @return CRC32 checksum
	 */
	uint32_t CrcCalc::calcCrc32Uint8(uint32_t crc, uint8_t data)
	{
		int32_t i;

		//This shift is done to make it compatible to the STM32 hardware CRC
		crc = crc ^ (static_cast<uint32_t>(data) << 24);

		for (i = 0; i < 8; i++)
		{
			if (crc & 0x80000000)
			{
				crc = (crc << 1) ^ polynom;
			}
			else
			{
				crc = (crc << 1);
			}
		}
		return (crc);
	}

	/**
	 * @brief Calculate CRC32 of one Byte
	 *
	 * This function calculates the CRC32 checksum of one 32bit value. It is only
	 * used internally as private function
	 *
	 * @param crc Actual CRC value
	 * @param byte Byte to use
	 * @return CRC32 checksum
	 */
	uint32_t CrcCalc::calcCrc32Uint32(uint32_t crc, uint32_t data)
	{
		int32_t i;

		crc = crc ^ data;

		for (i = 0; i < 32; i++)
		{
			if (crc & 0x80000000)
			{
				crc = (crc << 1) ^ polynom;
			}
			else
			{
				crc = (crc << 1);
			}
		}
		return(crc);
	}

}

/** @} */

