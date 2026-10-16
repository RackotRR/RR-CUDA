#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace RR::CUDA {

    constexpr cudaMemcpyKind ToDevice = cudaMemcpyHostToDevice;
    constexpr cudaMemcpyKind ToHost = cudaMemcpyDeviceToHost;
    constexpr cudaMemcpyKind DeviceToDevice = cudaMemcpyDeviceToDevice;

    inline std::string formatError(
        const std::string& func_name,
        cudaError_t error_code
    ) {
        return func_name + " error: " +
            cudaGetErrorName(error_code) +
            '(' + cudaGetErrorString(error_code) + ')';
    }

    inline void CuDeviceSync() {
        auto result = cudaDeviceSynchronize();

        if (result != cudaSuccess) {
            throw std::runtime_error{
                formatError(
                    "cudaDeviceSynchronize",
                    result
                )
            };
        }
    }

    template<typename FuncT>
    struct CuCallImpl {
        FuncT func;
        dim3 blocks_per_grid;
        dim3 threads_per_block;
        unsigned shared_mem_size;

        template<typename... Args>
        auto operator() (Args&&... args) {
            func<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(args...);
            auto result = cudaDeviceSynchronize();
            if (result != cudaSuccess) {
                throw std::runtime_error{
                    formatError(
                        "cuda invoke kernel",
                        result
                    )
                };
            }
        }
    };

    template<typename FuncT>
    CuCallImpl<FuncT> CuCall(
        FuncT func,
        const dim3& blocks_per_grid,
        const dim3& threads_per_block,
        const unsigned shared_mem_size = 0
    )
    {
        return CuCallImpl<FuncT>{ func, blocks_per_grid, threads_per_block, shared_mem_size };
    }

    class CuDarrayBase {
    public:
        static size_t get_total_allocated_mb() {
            return total_allocated / 1024 / 1024;
        }
    protected:
        static void add_alloc(size_t bytes) {
            total_allocated += bytes;
        }
        static void remove_alloc(size_t bytes) {
            total_allocated -= bytes;
        }
    private:
        inline static size_t total_allocated = 0;
    };

    template<typename T>
    class CuDarray : public CuDarrayBase {
    public:
        CuDarray() = default;
        CuDarray(size_t N_)
            : ptr{ alloc(N_) }
            , N{ N_ }
        {
            CuDarrayBase::add_alloc(get_allocated());
            set_zero();
        }
        CuDarray(const std::vector<T>& host)
            : ptr{ alloc(host.size()) }
            , N{ host.size() }
        {
            CuDarrayBase::add_alloc(get_allocated());
            copy_vector(host);
        }
        ~CuDarray() {
            cudaFree(ptr);
            CuDarrayBase::remove_alloc(get_allocated());
        }
        CuDarray(const CuDarray&) = delete;
        CuDarray(CuDarray&& other) noexcept
            : ptr{ std::exchange(other.ptr, nullptr) }
            , N{ std::exchange(other.N, 0) }
        {
        }
        auto& operator=(const CuDarray&) = delete;
        auto& operator=(CuDarray&& other) noexcept {
            this->ptr = std::exchange(other.ptr, nullptr);
            this->N = std::exchange(other.N, 0);
            return *this;
        }

        void set_zero() {
            auto result = cudaMemset(ptr, 0, get_allocated());
            if (result != cudaSuccess) {
                throw std::runtime_error{
                    formatError(
                        "cudaMemset",
                        result
                    )
                };
            }

            CuDeviceSync();
        }

        void copy_vector(const std::vector<T>& host) {
            if (host.size() != N) {
                throw std::runtime_error{ "CuDarray::copy_vector size mismatch" };
            }

            auto result = cudaMemcpy(ptr, host.data(), get_allocated(), ToDevice);
            if (result != cudaSuccess) {
                throw std::runtime_error{
                    formatError(
                        "cudaMemcpy to device",
                        result
                    )
                };
            }

            CuDeviceSync();
        }
        void to_vector(std::vector<T>& host) const {
            host.resize(N);
            auto result = cudaMemcpy(host.data(), ptr, get_allocated(), ToHost);
            if (result != cudaSuccess) {
                throw std::runtime_error{
                    formatError(
                        "cudaMemcpy to host",
                        result
                    )
                };
            }

            CuDeviceSync();
        }
        std::vector<T> to_vector() const {
            std::vector<T> host;
            to_vector(host);
            return host;
        }

        size_t get_allocated() const {
            return N * sizeof(T);
        }

        operator T* () const {
            return ptr;
        }

        void swap(CuDarray<T>& other) noexcept {
            std::swap(ptr, other.ptr);
            std::swap(N, other.N);
        }

    private:
        static T* alloc(size_t N) {
            T* ptr = nullptr;

            size_t need_mem = sizeof(T) * N;
            auto result = cudaMalloc(
                (void**)&ptr,
                need_mem
            );

            if (result != cudaSuccess) {
                throw std::runtime_error{
                    formatError(
                        "cudaMalloc",
                        result
                    )
                };
            }

            return ptr;
        }
    private:
        T* ptr = nullptr;
        size_t N = 0;
    };

    template<typename T>
    inline void swap(CuDarray<T>& left, CuDarray<T>& right) noexcept {
        left.swap(right);
    }

    class CuEvent {
    public:
        CuEvent() {
            cudaEventCreate(&event);
        }
        ~CuEvent() {
            cudaEventDestroy(event);
        }

        operator cudaEvent_t() const {
            return event;
        }

        CuEvent(const CuEvent&) = delete;
        CuEvent(CuEvent&&) = delete;
        auto& operator=(const CuEvent&) = delete;
        auto& operator=(CuEvent&&) = delete;
    private:
        cudaEvent_t event;
    };

    class CuTimer {
    public:
        void start() {
            cudaEventRecord(start_);
        }
        void stop() {
            cudaEventRecord(stop_);
            cudaEventSynchronize(stop_);
        }

        float elapsedMilliseconds() const {
            float ms = 0;
            auto result = cudaEventElapsedTime(&ms, start_, stop_);

            if (result != cudaSuccess) {
                throw std::runtime_error{
                    formatError(
                        "cudaEventElapsedTime",
                        result
                    )
                };
            }

            return ms;
        }

        float elapsedSeconds() const {
            return elapsedMilliseconds() * 0.001;
        }
    private:
        CuEvent start_, stop_;
    };

    template<typename T>
    void CuCopyFromTo(const CuDarray<T>& from, CuDarray<T>& to, int kind) {
        auto result = cudaMemcpy(
            to,
            from,
            from.get_allocated(),
            (cudaMemcpyKind)kind
        );
        CuDeviceSync();

        if (result != cudaSuccess) {
            throw std::runtime_error{
                formatError(
                    "cudaMemcpy",
                    result
                )
            };
        }
    }

    template<typename T>
    void CuCopyToSymbol(const T& from, T& to, int kind) {
        auto result = cudaMemcpyToSymbol(
            to,
            &from,
            sizeof(T),
            0,
            (cudaMemcpyKind)kind
        );
        CuDeviceSync();

        if (result != cudaSuccess) {
            throw std::runtime_error{
                formatError(
                    "cudaMemcpyToSymbol",
                    result
                )
            };
        }
    }

    inline std::vector<cudaDeviceProp> CuGetDevices() {
        int deviceCount = 0;
        cudaError_t result = cudaGetDeviceCount(&deviceCount);
        if (result != cudaSuccess) {
            throw std::runtime_error{
                formatError(
                    "cudaGetDeviceCount",
                    result
                )
            };
        }

        std::vector<cudaDeviceProp> devices(deviceCount);
        for (int i = 0; i < deviceCount; i++){
            cudaGetDeviceProperties(&devices[i], i);
        }
        return devices;
    }

}