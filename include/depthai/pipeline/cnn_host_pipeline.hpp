#pragma once

#include <unordered_map>
#include <vector>

#include "host_pipeline.hpp"
#include "../nnet/tensor_info.hpp"
#include "../nnet/nnet_packet.hpp"


class CNNHostPipeline
    : public HostPipeline
{
private:

    const std::string               cnn_result_stream_name = "metaout";

    const std::vector<TensorInfo>   _input_tensors_info;
    const std::vector<TensorInfo>   _output_tensors_info;


    std::list<std::shared_ptr<NNetPacket>> getConsumedNNetPackets();

public:
    CNNHostPipeline(const std::vector<TensorInfo>& input_tensors_info, const std::vector<TensorInfo>& output_tensors_info)
        : _input_tensors_info(input_tensors_info)
        , _output_tensors_info(output_tensors_info)
    {}
    virtual ~CNNHostPipeline() {}


    std::tuple<
        std::list<std::shared_ptr<NNetPacket>>,
        std::list<std::shared_ptr<HostDataPacket>>
    >
    getAvailableNNetAndDataPackets(bool blocking = false);


};
