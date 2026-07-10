#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cupti_profiler_host.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct MetricValue {
    std::string name;
    double value = 0.0;
};

struct GpuCounterCollector {
    CUcontext context = nullptr;
    CUdevice device = 0;
    CUpti_Profiler_Host_Object* host = nullptr;
    CUpti_RangeProfiler_Object* profiler = nullptr;
    std::vector<std::string> metric_names;
    std::vector<const char*> metric_name_ptrs;
    std::vector<std::uint8_t> counter_availability;
    std::vector<std::uint8_t> config_image;
    std::vector<std::uint8_t> counter_data_image;
    std::vector<MetricValue> values;
    bool all_passes_submitted = false;
    bool started = false;
};

void set_error(char* error, std::size_t error_len, const std::string& message) {
    if (error == nullptr || error_len == 0) {
        return;
    }
    const std::size_t copy_len = std::min(error_len - 1, message.size());
    std::memcpy(error, message.data(), copy_len);
    error[copy_len] = '\0';
}

std::string cupti_error(CUptiResult result, const char* label) {
    const char* text = nullptr;
    cuptiGetResultString(result, &text);
    return std::string(label) + ": " + (text == nullptr ? "unknown CUPTI error" : text);
}

std::string cuda_error(CUresult result, const char* label) {
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    return std::string(label) + ": " + (text == nullptr ? "unknown CUDA driver error" : text);
}

int fail(char* error, std::size_t error_len, const std::string& message) {
    set_error(error, error_len, message);
    return 1;
}

int check_cupti(CUptiResult result, const char* label, char* error, std::size_t error_len) {
    if (result == CUPTI_SUCCESS) {
        return 0;
    }
    return fail(error, error_len, cupti_error(result, label));
}

int check_cuda(CUresult result, const char* label, char* error, std::size_t error_len) {
    if (result == CUDA_SUCCESS) {
        return 0;
    }
    return fail(error, error_len, cuda_error(result, label));
}

void destroy_collector(GpuCounterCollector* collector) {
    if (collector == nullptr) {
        return;
    }
    if (collector->profiler != nullptr) {
        CUpti_RangeProfiler_Disable_Params params = {CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
        params.pRangeProfilerObject = collector->profiler;
        cuptiRangeProfilerDisable(&params);
    }
    if (collector->host != nullptr) {
        CUpti_Profiler_Host_Deinitialize_Params params = {CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        params.pHostObject = collector->host;
        cuptiProfilerHostDeinitialize(&params);
    }
    CUpti_Profiler_DeInitialize_Params deinit = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    cuptiProfilerDeInitialize(&deinit);
    delete collector;
}

int initialize_host(GpuCounterCollector* collector, char* error, std::size_t error_len) {
    CUpti_Device_GetChipName_Params chip_params = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip_params.deviceIndex = static_cast<std::size_t>(collector->device);
    if (int rc = check_cupti(cuptiDeviceGetChipName(&chip_params), "cuptiDeviceGetChipName", error, error_len)) {
        return rc;
    }

    CUpti_Profiler_GetCounterAvailability_Params availability_size = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    availability_size.ctx = collector->context;
    if (int rc = check_cupti(cuptiProfilerGetCounterAvailability(&availability_size), "cuptiProfilerGetCounterAvailability(size)", error, error_len)) {
        return rc;
    }
    collector->counter_availability.resize(availability_size.counterAvailabilityImageSize);
    CUpti_Profiler_GetCounterAvailability_Params availability = {CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    availability.ctx = collector->context;
    availability.pCounterAvailabilityImage = collector->counter_availability.data();
    availability.counterAvailabilityImageSize = collector->counter_availability.size();
    if (int rc = check_cupti(cuptiProfilerGetCounterAvailability(&availability), "cuptiProfilerGetCounterAvailability", error, error_len)) {
        return rc;
    }

    CUpti_Profiler_Host_Initialize_Params host_init = {CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    host_init.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    host_init.pChipName = chip_params.pChipName;
    host_init.pCounterAvailabilityImage = collector->counter_availability.data();
    if (int rc = check_cupti(cuptiProfilerHostInitialize(&host_init), "cuptiProfilerHostInitialize", error, error_len)) {
        return rc;
    }
    collector->host = host_init.pHostObject;
    return 0;
}

int create_config_image(GpuCounterCollector* collector, char* error, std::size_t error_len) {
    CUpti_Profiler_Host_ConfigAddMetrics_Params add = {CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    add.pHostObject = collector->host;
    add.ppMetricNames = collector->metric_name_ptrs.data();
    add.numMetrics = collector->metric_name_ptrs.size();
    if (int rc = check_cupti(cuptiProfilerHostConfigAddMetrics(&add), "cuptiProfilerHostConfigAddMetrics", error, error_len)) {
        return rc;
    }

    CUpti_Profiler_Host_GetConfigImageSize_Params size = {CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    size.pHostObject = collector->host;
    if (int rc = check_cupti(cuptiProfilerHostGetConfigImageSize(&size), "cuptiProfilerHostGetConfigImageSize", error, error_len)) {
        return rc;
    }
    collector->config_image.resize(size.configImageSize);

    CUpti_Profiler_Host_GetConfigImage_Params image = {CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    image.pHostObject = collector->host;
    image.pConfigImage = collector->config_image.data();
    image.configImageSize = collector->config_image.size();
    return check_cupti(cuptiProfilerHostGetConfigImage(&image), "cuptiProfilerHostGetConfigImage", error, error_len);
}

int enable_profiler(GpuCounterCollector* collector, char* error, std::size_t error_len) {
    CUpti_RangeProfiler_Enable_Params enable = {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    enable.ctx = collector->context;
    if (int rc = check_cupti(cuptiRangeProfilerEnable(&enable), "cuptiRangeProfilerEnable", error, error_len)) {
        return rc;
    }
    collector->profiler = enable.pRangeProfilerObject;

    CUpti_RangeProfiler_GetCounterDataSize_Params data_size = {CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    data_size.pRangeProfilerObject = collector->profiler;
    data_size.pMetricNames = collector->metric_name_ptrs.data();
    data_size.numMetrics = collector->metric_name_ptrs.size();
    data_size.maxNumOfRanges = 1;
    data_size.maxNumRangeTreeNodes = 1;
    if (int rc = check_cupti(cuptiRangeProfilerGetCounterDataSize(&data_size), "cuptiRangeProfilerGetCounterDataSize", error, error_len)) {
        return rc;
    }
    collector->counter_data_image.resize(data_size.counterDataSize, 0);

    CUpti_RangeProfiler_CounterDataImage_Initialize_Params init = {CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init.pRangeProfilerObject = collector->profiler;
    init.pCounterData = collector->counter_data_image.data();
    init.counterDataSize = collector->counter_data_image.size();
    if (int rc = check_cupti(cuptiRangeProfilerCounterDataImageInitialize(&init), "cuptiRangeProfilerCounterDataImageInitialize", error, error_len)) {
        return rc;
    }

    CUpti_RangeProfiler_SetConfig_Params config = {CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    config.pRangeProfilerObject = collector->profiler;
    config.pConfig = collector->config_image.data();
    config.configSize = collector->config_image.size();
    config.pCounterDataImage = collector->counter_data_image.data();
    config.counterDataImageSize = collector->counter_data_image.size();
    config.maxRangesPerPass = 1;
    config.numNestingLevels = 1;
    config.minNestingLevel = 1;
    config.passIndex = 0;
    config.targetNestingLevel = 1;
    config.range = CUPTI_UserRange;
    config.replayMode = CUPTI_UserReplay;
    return check_cupti(cuptiRangeProfilerSetConfig(&config), "cuptiRangeProfilerSetConfig", error, error_len);
}

}  // namespace

extern "C" int infer_gpu_counter_create(const char* const* metric_names,
                                               std::size_t metric_count,
                                               void** out_handle,
                                               char* error,
                                               std::size_t error_len) {
    if (out_handle == nullptr) {
        return fail(error, error_len, "out_handle is null");
    }
    *out_handle = nullptr;
    if (metric_names == nullptr || metric_count == 0) {
        return fail(error, error_len, "no GPU counter metrics requested");
    }

    cudaFree(nullptr);

    auto* collector = new GpuCounterCollector();
    for (std::size_t i = 0; i < metric_count; ++i) {
        collector->metric_names.emplace_back(metric_names[i]);
    }
    collector->metric_name_ptrs.reserve(collector->metric_names.size());
    for (const auto& metric : collector->metric_names) {
        collector->metric_name_ptrs.push_back(metric.c_str());
    }

    CUpti_Profiler_Initialize_Params init = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    if (int rc = check_cupti(cuptiProfilerInitialize(&init), "cuptiProfilerInitialize", error, error_len)) {
        delete collector;
        return rc;
    }
    if (int rc = check_cuda(cuCtxGetCurrent(&collector->context), "cuCtxGetCurrent", error, error_len)) {
        destroy_collector(collector);
        return rc;
    }
    if (collector->context == nullptr) {
        destroy_collector(collector);
        return fail(error, error_len, "no current CUDA context");
    }
    if (int rc = check_cuda(cuCtxGetDevice(&collector->device), "cuCtxGetDevice", error, error_len)) {
        destroy_collector(collector);
        return rc;
    }
    if (int rc = initialize_host(collector, error, error_len)) {
        destroy_collector(collector);
        return rc;
    }
    if (int rc = create_config_image(collector, error, error_len)) {
        destroy_collector(collector);
        return rc;
    }
    if (int rc = enable_profiler(collector, error, error_len)) {
        destroy_collector(collector);
        return rc;
    }

    *out_handle = collector;
    return 0;
}

extern "C" int infer_gpu_counter_begin(void* handle,
                                              const char* range_name,
                                              char* error,
                                              std::size_t error_len) {
    auto* collector = static_cast<GpuCounterCollector*>(handle);
    if (collector == nullptr) {
        return fail(error, error_len, "GPU counter collector is null");
    }
    CUpti_RangeProfiler_Start_Params start = {CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
    start.pRangeProfilerObject = collector->profiler;
    if (int rc = check_cupti(cuptiRangeProfilerStart(&start), "cuptiRangeProfilerStart", error, error_len)) {
        return rc;
    }
    CUpti_RangeProfiler_PushRange_Params push = {CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE};
    push.pRangeProfilerObject = collector->profiler;
    push.pRangeName = range_name == nullptr ? "infer" : range_name;
    if (int rc = check_cupti(cuptiRangeProfilerPushRange(&push), "cuptiRangeProfilerPushRange", error, error_len)) {
        return rc;
    }
    collector->started = true;
    return 0;
}

extern "C" int infer_gpu_counter_end(void* handle,
                                            int* all_passes_submitted,
                                            char* error,
                                            std::size_t error_len) {
    auto* collector = static_cast<GpuCounterCollector*>(handle);
    if (collector == nullptr) {
        return fail(error, error_len, "GPU counter collector is null");
    }
    if (collector->started) {
        CUpti_RangeProfiler_PopRange_Params pop = {CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE};
        pop.pRangeProfilerObject = collector->profiler;
        if (int rc = check_cupti(cuptiRangeProfilerPopRange(&pop), "cuptiRangeProfilerPopRange", error, error_len)) {
            return rc;
        }
        collector->started = false;
    }
    CUpti_RangeProfiler_Stop_Params stop = {CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
    stop.pRangeProfilerObject = collector->profiler;
    if (int rc = check_cupti(cuptiRangeProfilerStop(&stop), "cuptiRangeProfilerStop", error, error_len)) {
        return rc;
    }
    collector->all_passes_submitted = stop.isAllPassSubmitted;
    if (all_passes_submitted != nullptr) {
        *all_passes_submitted = collector->all_passes_submitted ? 1 : 0;
    }
    return 0;
}

extern "C" int infer_gpu_counter_decode(void* handle, char* error, std::size_t error_len) {
    auto* collector = static_cast<GpuCounterCollector*>(handle);
    if (collector == nullptr) {
        return fail(error, error_len, "GPU counter collector is null");
    }
    CUpti_RangeProfiler_DecodeData_Params decode = {CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decode.pRangeProfilerObject = collector->profiler;
    if (int rc = check_cupti(cuptiRangeProfilerDecodeData(&decode), "cuptiRangeProfilerDecodeData", error, error_len)) {
        return rc;
    }
    CUpti_RangeProfiler_GetCounterDataInfo_Params info = {CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE};
    info.pCounterDataImage = collector->counter_data_image.data();
    info.counterDataImageSize = collector->counter_data_image.size();
    if (int rc = check_cupti(cuptiRangeProfilerGetCounterDataInfo(&info), "cuptiRangeProfilerGetCounterDataInfo", error, error_len)) {
        return rc;
    }
    if (info.numTotalRanges == 0) {
        return fail(error, error_len, "GPU counter data contains no ranges");
    }

    collector->values.clear();
    std::vector<double> metric_values(collector->metric_name_ptrs.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params eval = {CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    eval.pHostObject = collector->host;
    eval.pCounterDataImage = collector->counter_data_image.data();
    eval.counterDataImageSize = collector->counter_data_image.size();
    eval.ppMetricNames = collector->metric_name_ptrs.data();
    eval.numMetrics = collector->metric_name_ptrs.size();
    eval.rangeIndex = 0;
    eval.pMetricValues = metric_values.data();
    if (int rc = check_cupti(cuptiProfilerHostEvaluateToGpuValues(&eval), "cuptiProfilerHostEvaluateToGpuValues", error, error_len)) {
        return rc;
    }
    for (std::size_t i = 0; i < collector->metric_names.size(); ++i) {
        collector->values.push_back(MetricValue{collector->metric_names[i], metric_values[i]});
    }
    return 0;
}

extern "C" std::size_t infer_gpu_counter_value_count(void* handle) {
    auto* collector = static_cast<GpuCounterCollector*>(handle);
    return collector == nullptr ? 0 : collector->values.size();
}

extern "C" int infer_gpu_counter_value(void* handle,
                                              std::size_t index,
                                              const char** name,
                                              double* value) {
    auto* collector = static_cast<GpuCounterCollector*>(handle);
    if (collector == nullptr || index >= collector->values.size() || name == nullptr || value == nullptr) {
        return 1;
    }
    *name = collector->values[index].name.c_str();
    *value = collector->values[index].value;
    return 0;
}

extern "C" void infer_gpu_counter_destroy(void* handle) {
    destroy_collector(static_cast<GpuCounterCollector*>(handle));
}
