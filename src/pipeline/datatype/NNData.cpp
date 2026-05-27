#include "depthai/pipeline/datatype/NNData.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include "depthai-shared/datatype/RawNNData.hpp"
#include "depthai/pipeline/datatype/ADatatype.hpp"

// Inlined IEEE half<->float conversions (originally from fp16/fp16.h by Maratyszcza, MIT).
// Vendored locally so depthai-core does not depend on the FP16 Hunter package.
namespace {

inline float fp32_from_bits(std::uint32_t w) {
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

inline std::uint32_t fp32_to_bits(float f) {
    std::uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    return w;
}

inline float fp16_ieee_to_fp32_value(std::uint16_t h) {
    const std::uint32_t w = static_cast<std::uint32_t>(h) << 16;
    const std::uint32_t sign = w & UINT32_C(0x80000000);
    const std::uint32_t two_w = w + w;

    const std::uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const std::uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const std::uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const std::uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

inline std::uint16_t fp16_ieee_from_fp32_value(float f) {
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
    float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;

    const std::uint32_t w = fp32_to_bits(f);
    const std::uint32_t shl1_w = w + w;
    const std::uint32_t sign = w & UINT32_C(0x80000000);
    std::uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const std::uint32_t bits = fp32_to_bits(base);
    const std::uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const std::uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const std::uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

}  // namespace

namespace dai {

NNData::NNData() : Buffer(std::make_shared<RawNNData>()), rawNn(*dynamic_cast<RawNNData*>(raw.get())) {}
NNData::NNData(std::shared_ptr<RawNNData> ptr) : Buffer(ptr), rawNn(*ptr.get()) {}

static std::size_t sizeofTensorInfoDataType(TensorInfo::DataType type) {
    switch(type) {
        case TensorInfo::DataType::FP16:
            return sizeof(uint16_t);
        case TensorInfo::DataType::FP32:
            return sizeof(float);
        case TensorInfo::DataType::I8:
            return sizeof(int8_t);
        case TensorInfo::DataType::INT:
            return sizeof(int32_t);
        case TensorInfo::DataType::U8F:
            return sizeof(uint8_t);
        default:
            // invalid type, shouldn't happen
            assert(0);
            return 0;
    }
}

static std::size_t getTensorDataSize(const TensorInfo& tensor) {
    return tensor.dims[0] * tensor.strides[0];
}

std::shared_ptr<RawBuffer> NNData::serialize() const {
    // get data from u8Data and fp16Data and place properly into the underlying raw buffer
    rawNn.tensors = {};
    rawNn.data.clear();

    // U8 tensors
    for(const auto& kv : u8Data) {
        const auto dataType = TensorInfo::DataType::U8F;
        const auto dataSize = kv.second.size() * sizeofTensorInfoDataType(dataType);

        // First add any required padding bytes
        // calculate how many alignment bytes to add for next tensor
        size_t remainder = (rawNn.data.end() - rawNn.data.begin()) % DATA_ALIGNMENT;
        if(remainder > 0) {
            rawNn.data.insert(rawNn.data.end(), DATA_ALIGNMENT - remainder, 0);
        }

        // Then get offset to beginning of data
        size_t offset = rawNn.data.end() - rawNn.data.begin();

        const auto* data = reinterpret_cast<const std::uint8_t*>(kv.second.data());
        rawNn.data.insert(rawNn.data.end(), data, data + dataSize);

        // Add entry in tensors
        // TODO(themarpe) - refactor with proper way off specifying tensors
        TensorInfo info;
        info.dataType = dataType;
        info.numDimensions = 1;
        info.dims.push_back(kv.second.size());
        info.strides.push_back(sizeofTensorInfoDataType(dataType));
        info.name = kv.first;
        info.offset = static_cast<unsigned int>(offset);
        rawNn.tensors.push_back(info);
    }

    // FP16 tensors
    for(const auto& kv : fp16Data) {
        const auto dataType = TensorInfo::DataType::FP16;
        const auto dataSize = kv.second.size() * sizeofTensorInfoDataType(dataType);

        // First add any required padding bytes
        // calculate how many alignment bytes to add for next tensor
        size_t remainder = (rawNn.data.end() - rawNn.data.begin()) % DATA_ALIGNMENT;
        if(remainder > 0) {
            rawNn.data.insert(rawNn.data.end(), DATA_ALIGNMENT - remainder, 0);
        }

        // Then get offset to beginning of data
        size_t offset = rawNn.data.end() - rawNn.data.begin();

        const auto* data = reinterpret_cast<const std::uint8_t*>(kv.second.data());
        rawNn.data.insert(rawNn.data.end(), data, data + dataSize);

        // Add entry in tensors
        // TODO(themarpe) - refactor with proper way off specifying tensors
        TensorInfo info;
        info.dataType = dataType;
        info.numDimensions = 1;
        info.dims.push_back(kv.second.size());
        info.strides.push_back(sizeofTensorInfoDataType(dataType));
        info.name = kv.first;
        info.offset = static_cast<unsigned int>(offset);
        rawNn.tensors.push_back(info);
    }

    return raw;
}

// setters
// uint8_t
NNData& NNData::setLayer(const std::string& name, std::vector<std::uint8_t> data) {
    u8Data[name] = std::move(data);
    return *this;
}
NNData& NNData::setLayer(const std::string& name, const std::vector<int>& data) {
    u8Data[name] = std::vector<std::uint8_t>(data.size());
    for(unsigned i = 0; i < data.size(); i++) {
        u8Data[name][i] = static_cast<std::uint8_t>(data[i]);
    }
    return *this;
}

// fp16
NNData& NNData::setLayer(const std::string& name, std::vector<float> data) {
    fp16Data[name] = std::vector<std::uint16_t>(data.size());
    for(unsigned i = 0; i < data.size(); i++) {
        fp16Data[name][i] = fp16_ieee_from_fp32_value(data[i]);
    }
    return *this;
}
NNData& NNData::setLayer(const std::string& name, std::vector<double> data) {
    fp16Data[name] = std::vector<std::uint16_t>(data.size());
    for(unsigned i = 0; i < data.size(); i++) {
        fp16Data[name][i] = fp16_ieee_from_fp32_value(static_cast<float>(data[i]));
    }
    return *this;
}

// getters
std::vector<std::string> NNData::getAllLayerNames() const {
    std::vector<std::string> names;
    for(const auto& t : rawNn.tensors) {
        names.push_back(t.name);
    }
    return names;
}

std::vector<TensorInfo> NNData::getAllLayers() const {
    return rawNn.tensors;
}

bool NNData::getLayer(const std::string& name, TensorInfo& tensor) const {
    for(const auto& t : rawNn.tensors) {
        if(t.name == name) {
            tensor = t;
            return true;
        }
    }
    return false;
}

bool NNData::hasLayer(const std::string& name) const {
    for(const auto& tensor : rawNn.tensors) {
        if(tensor.name == name) return true;
    }
    return false;
}

bool NNData::getLayerDatatype(const std::string& name, TensorInfo::DataType& datatype) const {
    TensorInfo tensor;
    if(getLayer(name, tensor)) {
        datatype = tensor.dataType;
        return true;
    }
    return false;
}

// uint8
std::vector<std::uint8_t> NNData::getLayerUInt8(const std::string& name) const {
    // std::vector<std::uint8_t> data;
    // find layer name and its offset
    TensorInfo tensor;
    if(getLayer(name, tensor)) {
        if(tensor.dataType == TensorInfo::DataType::U8F) {
            // Total data size = last dimension * last stride
            if(tensor.numDimensions > 0) {
                size_t size = getTensorDataSize(tensor);
                auto beg = rawNn.data.begin() + tensor.offset;
                auto end = beg + size;
                return {beg, end};
            }
        }
    }
    return {};
}

// int32_t
std::vector<std::int32_t> NNData::getLayerInt32(const std::string& name) const {
    // find layer name and its offset
    TensorInfo tensor;
    if(getLayer(name, tensor)) {
        if(tensor.dataType == TensorInfo::DataType::INT) {
            // Total data size = last dimension * last stride
            if(tensor.numDimensions > 0) {
                size_t size = getTensorDataSize(tensor);
                std::size_t numElements = size / sizeof(std::int32_t);  // FP16

                std::vector<std::int32_t> data;
                data.reserve(numElements);
                auto* pInt32Data = reinterpret_cast<std::int32_t*>(&rawNn.data[tensor.offset]);
                for(std::size_t i = 0; i < numElements; i++) {
                    data.push_back(pInt32Data[i]);
                }
                return data;
            }
        }
    }
    return {};
}

// fp16
std::vector<float> NNData::getLayerFp16(const std::string& name) const {
    // find layer name and its offset
    TensorInfo tensor;
    if(getLayer(name, tensor)) {
        if(tensor.dataType == TensorInfo::DataType::FP16) {
            // Total data size = last dimension * last stride
            if(tensor.numDimensions > 0) {
                std::size_t size = getTensorDataSize(tensor);
                std::size_t numElements = size / 2;  // FP16

                std::vector<float> data;
                data.reserve(numElements);
                auto* pFp16Data = reinterpret_cast<std::uint16_t*>(&rawNn.data[tensor.offset]);
                for(std::size_t i = 0; i < numElements; i++) {
                    data.push_back(fp16_ieee_to_fp32_value(pFp16Data[i]));
                }
                return data;
            }
        }
    }
    return {};
}

// uint8
std::vector<std::uint8_t> NNData::getFirstLayerUInt8() const {
    // find layer name and its offset
    if(!rawNn.tensors.empty()) {
        return getLayerUInt8(rawNn.tensors[0].name);
    }

    return {};
}

// fp16
std::vector<float> NNData::getFirstLayerFp16() const {
    // find layer name and its offset
    if(!rawNn.tensors.empty()) {
        return getLayerFp16(rawNn.tensors[0].name);
    }
    return {};
}

// int32
std::vector<std::int32_t> NNData::getFirstLayerInt32() const {
    // find layer name and its offset
    if(!rawNn.tensors.empty()) {
        return getLayerInt32(rawNn.tensors[0].name);
    }
    return {};
}

// setters
NNData& NNData::setTimestamp(std::chrono::time_point<std::chrono::steady_clock, std::chrono::steady_clock::duration> tp) {
    // Set timestamp from timepoint
    return static_cast<NNData&>(Buffer::setTimestamp(tp));
}
NNData& NNData::setTimestampDevice(std::chrono::time_point<std::chrono::steady_clock, std::chrono::steady_clock::duration> tp) {
    // Set timestamp from timepoint
    return static_cast<NNData&>(Buffer::setTimestampDevice(tp));
}
NNData& NNData::setSequenceNum(int64_t sequenceNum) {
    return static_cast<NNData&>(Buffer::setSequenceNum(sequenceNum));
}

}  // namespace dai
