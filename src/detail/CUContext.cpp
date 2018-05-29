#include <cuda.h>

#include "detail/CUContext.h"
#include "detail/utils.h"

namespace NVVL {
namespace detail {

CUContext::CUContext() : context_{0}, initialized_{false} {
}

CUContext::CUContext(CUdevice device, unsigned int flags)
    : device_{device}, context_{0}, initialized_{false} {
    //cucall(cuInit(0));
    if (!cucall(cuDevicePrimaryCtxRetain(&context_, device))) {
        throw std::runtime_error("cuDevicePrimaryCtxRetain failed, can't go forward without a context");
    }
    push(__FILE__, __LINE__);
    CUdevice dev;
    if (!cucall(cuCtxGetDevice(&dev))) {
        throw std::runtime_error("Unable to get device");
    }
    initialized_ = true;
    cucall(cuCtxSynchronize());
}

CUContext::CUContext(CUcontext ctx)
    : context_{ctx}, initialized_{true} {
}

CUContext::~CUContext() {
    if (initialized_) {
        // cuCtxPopCurrent?
        cucall(cuDevicePrimaryCtxRelease(device_));
    }
}

CUContext::CUContext(CUContext&& other)
    : device_{other.device_}, context_{other.context_},
      initialized_{other.initialized_} {
    other.device_ = 0;
    other.context_ = 0;
    other.initialized_ = false;
}

CUContext& CUContext::operator=(CUContext&& other) {
    if (initialized_) {
        cucall(cuCtxDestroy(context_));
    }
    device_ = other.device_;
    context_ = other.context_;
    initialized_ = other.initialized_;
    other.device_ = 0;
    other.context_ = 0;
    other.initialized_ = false;
    return *this;
}

void CUContext::push(std::string file, int line) const {
    CUcontext current;
    if (!cucall(cuCtxGetCurrent(&current))) {
        throw std::runtime_error("Unable to get current context");
    }

    CUdevice device;
    cuCtxGetDevice(&device);
    std::cout << "current device: " << device << std::endl;

    if (current != context_) {
        if (!cucall(cuCtxPushCurrent(context_))) {
            throw std::runtime_error("Unable to push current context");
        }
        std::cout << "cuCtxPushCurrent: " << file << ":" << line << std::endl;
    }
}

bool CUContext::initialized() const {
    return initialized_;
}

CUContext::operator CUcontext() const {
    return context_;
}


}
}
