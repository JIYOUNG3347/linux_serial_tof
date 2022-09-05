#pragma once
/**
 * Copyright (C) 2017 Espros Photonics Corporation
 *
 * @defgroup crc CRC
 *
 * @brief CRC calculation
 * @ingroup communication
 *
 * @{
 */
#ifndef BASE_CRC_CRC_H_
#define BASE_CRC_CRC_H_

#include <stdint.h>

namespace ComLib
{

    //! CRC calculation
    /*!
     * Class for the CRC calculation in software (no hardware assistance)
     *
     * The calculation of CRC can be found on wikipedia. It is here implemented straight forward.
     */
    class CrcCalc
    {
    public:
        static uint32_t calcCrc32(const uint8_t* data, const uint32_t size);
        static uint32_t calcCrc32(const uint32_t* data, const uint32_t size);
        static uint32_t calcCrc32_32(const uint8_t* data, const uint32_t size);

    private:
        static uint32_t calcCrc32Uint8(uint32_t crc, uint8_t data);
        static uint32_t calcCrc32Uint32(uint32_t crc, uint32_t data);
    };

#endif /* BASE_CRC_CRC_H_ */

}

/** @} */
