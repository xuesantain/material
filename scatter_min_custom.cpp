#include "kernel_operator.h"

constexpr int32_t N = 4096; // 输入长度
constexpr int32_t K = 1024; // 输出桶数
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = N / USE_CORE_NUM;
constexpr int32_t TILE_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM;

class KernelScatterMin {
public:
    __aicore__ inline KernelScatterMin() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR out)
    {
        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t start = blockIdx * BLOCK_LENGTH;
        srcGm.SetGlobalBuffer((__gm__ float *)src + start, BLOCK_LENGTH);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)index + start, BLOCK_LENGTH);
        outGm.SetGlobalBuffer((__gm__ float *)out, K);
        pipe.InitBuffer(inQueueSrc, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueueIdx, BUFFER_NUM, TILE_LENGTH * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        float local_min[K];
        for (int i = 0; i < K; ++i) local_min[i] = 1e30f; // +inf

        for (int t = 0; t < TILE_NUM; ++t) {
            AscendC::LocalTensor<float> srcLocal = inQueueSrc.AllocTensor<float>();
            AscendC::LocalTensor<int32_t> idxLocal = inQueueIdx.AllocTensor<int32_t>();
            AscendC::DataCopy(srcLocal, srcGm[t * TILE_LENGTH], TILE_LENGTH);
            AscendC::DataCopy(idxLocal, idxGm[t * TILE_LENGTH], TILE_LENGTH);
            for (int i = 0; i < TILE_LENGTH; ++i) {
                float val = srcLocal.GetValue(i);
                int idx = idxLocal.GetValue(i);
                if (idx >= 0 && idx < K) {
                    if (val < local_min[idx]) local_min[idx] = val;
                }
            }
            inQueueSrc.FreeTensor(srcLocal);
            inQueueIdx.FreeTensor(idxLocal);
        }
        int32_t blockIdx = AscendC::GetBlockIdx();
        int32_t base = blockIdx * K;
        for (int i = 0; i < K; ++i) {
            outGm.SetValue(base + i, local_min[i]);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueSrc, inQueueIdx;
    AscendC::GlobalTensor<float> srcGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> outGm;
};

extern "C" __global__ __aicore__ void scatter_min_custom(GM_ADDR src, GM_ADDR index, GM_ADDR out)
{
    KernelScatterMin op;
    op.Init(src, index, out);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void scatter_min_custom_do(uint32_t blockDim, void *stream, uint8_t *src, uint8_t *index, uint8_t *out)
{
    scatter_min_custom<<<blockDim, nullptr, stream>>>(src, index, out);
}
#endif 