# FUDA_HGEMM

CUDA Half-Precision GEMM (FP16 Matrix Multiply) 实现，支持扩展优化。

## 项目简介

本项目实现了基础的 HGEMM 算子，可用于学习和实践 CUDA 优化技术。项目结构清晰，便于添加新的 kernel 实现。

## 特性

- ✅ 基础 FP16 GEMM kernel 实现
- ✅ 可扩展的测试框架
- ✅ CMake 构建系统
- ✅ 支持 RTX 5090 Laptop (Blackwell 架构)
- ✅ 自动性能基准测试
- ✅ 与 cuBLAS 结果对比验证

## 快速开始

### 环境要求

- NVIDIA GPU (Blackwell/Ada/Ampere 及以上)
- CUDA Toolkit >= 12.0
- CMake >= 3.18
- C++17 编译器

### 编译和运行

```bash
# 创建构建目录
mkdir build && cd build

# 配置 (RTX 5090 Laptop 用户)
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build . -j$(nproc)

# 运行测试
./bin/test_hgemm_native
```

详细的编译说明请参考 [BUILD.md](BUILD.md)。

## 项目结构

```
FUDA_HGEMM/
├── cuda/           # CUDA kernel 实现
│   └── native/     # 原生实现
│       └── hgemm.cu
├── test/           # 测试程序
│   └── test_hgemm.cu
├── utils/          # 测试工具函数
│   └── test_utils.h
├── CMakeLists.txt  # CMake 构建配置
├── BUILD.md        # 详细编译指南
└── README.md       # 本文件
```

## 添加新 Kernel

1. 在 `cuda/native/` 目录下创建新的 `.cu` 文件
2. 在 `test/test_hgemm.cu` 中添加测试调用
3. 如果需要独立测试，在 `CMakeLists.txt` 中添加新的可执行文件

参考 [BUILD.md](BUILD.md) 中的"添加新的 Kernel"章节。

## 性能分析

使用 Nsight Compute 分析性能：

```bash
ncu --set full -o profile ./bin/test_hgemm_native
ncu-ui profile.ncu-rep
```

## 参考文献

- [HGEMM 优化指南](../HGEMM/HGEMM优化指南.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## 许可证

本项目仅供学习参考使用。
