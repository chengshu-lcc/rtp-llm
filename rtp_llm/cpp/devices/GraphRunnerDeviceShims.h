#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#define GRAPH_DEVICE_TYPE c10::DeviceType::HIP
#define GRAPH_GET_STREAM() at::hip::getCurrentHIPStream()
#else
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#define GRAPH_DEVICE_TYPE c10::DeviceType::CUDA
#define GRAPH_GET_STREAM() at::cuda::getCurrentCUDAStream()
#endif

namespace py = pybind11;

namespace rtp_llm {
#if USING_ROCM
namespace rocm {
void setHipGraphCaptureEnabled(bool enabled);
}
#endif
namespace graph_runner {

struct GraphNcclCaptureContext {
    int64_t comm_handle{0};
    int     rank{0};
    int     world_size{1};
};

inline void register_graph_capture_nccl_comm(void* nccl_comm, int world_size, int rank) {
#if USING_ROCM
    if (nccl_comm == nullptr || world_size <= 1) {
        return;
    }
    try {
        py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
        collective_torch.attr("set_hipgraph_capture_nccl_comm")(reinterpret_cast<int64_t>(nccl_comm), world_size, rank);
        RTP_LLM_LOG_INFO("Registered NCCL comm for graph capture (rank=%d, world_size=%d)", rank, world_size);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to register NCCL comm: %s", e.what());
    }
#else
    (void)nccl_comm;
    (void)world_size;
    (void)rank;
#endif
}

inline void enter_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    rocm::setHipGraphCaptureEnabled(true);
    try {
        py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
        if (ctx && ctx->comm_handle != 0) {
            collective_torch.attr("enter_hipgraph_capture_mode")(ctx->comm_handle, ctx->world_size, ctx->rank);
        } else {
            collective_torch.attr("enter_hipgraph_capture_mode")(0, 0, 0);
        }
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to enter graph capture mode: %s", e.what());
    }
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = true;
#endif
}

inline void exit_graph_capture(GraphNcclCaptureContext* ctx) {
#if USING_ROCM
    rocm::setHipGraphCaptureEnabled(false);
    try {
        py::module_ collective_torch = py::module_::import("rtp_llm.models_py.distributed.collective_torch");
        collective_torch.attr("exit_hipgraph_capture_mode")();
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_WARNING("Failed to exit graph capture mode: %s", e.what());
    }
#else
    (void)ctx;
    CaptureCheck::in_cuda_graph_capture = false;
#endif
}

}  // namespace graph_runner
}  // namespace rtp_llm
