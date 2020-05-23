# gpu-check

## GPU and driver information
GPU check for usage with tensorflow, keras and pytorch
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.82       Driver Version: 440.82       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 206...  Off  | 00000000:08:00.0  On |                  N/A |
|  0%   46C    P8    10W / 184W |    415MiB /  7979MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```
## CUDNN information
```
$ cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 5

```
## CUDA information
```
$ nvcc  --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```
## Tensorflow version
```
$ python3 -c 'import tensorflow as tf; print(tf.__version__)'  
2.2.0
```
## Pytorch check: OK
## Tensorflow 2.2 check: NOK
```
[I 15:12:08.582 NotebookApp] Kernel started: 548084d3-0e9c-425d-9ac8-e23d53e4cbb2
2020-05-23 15:12:11.446805: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 15:12:11.470046: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 3792815000 Hz
2020-05-23 15:12:11.470429: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f7da8000b20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 15:12:11.470446: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-05-23 15:12:11.471702: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-05-23 15:12:11.599639: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-05-23 15:12:11.600089: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x54e3a60 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-05-23 15:12:11.600121: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2060 SUPER, Compute Capability 7.5
2020-05-23 15:12:11.600319: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-05-23 15:12:11.600684: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:08:00.0 name: GeForce RTX 2060 SUPER computeCapability: 7.5
coreClock: 1.71GHz coreCount: 34 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
2020-05-23 15:12:11.600798: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-10.2/lib64:/usr/lib/x86_64-linux-gnu/:
2020-05-23 15:12:11.601711: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-05-23 15:12:11.602750: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-05-23 15:12:11.602896: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-05-23 15:12:11.603837: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-05-23 15:12:11.604395: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-05-23 15:12:11.606435: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-23 15:12:11.606447: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-05-23 15:12:11.606463: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-23 15:12:11.606469: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
2020-05-23 15:12:11.606474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
2020-05-23 15:12:11.607493: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-05-23 15:12:11.607855: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:08:00.0 name: GeForce RTX 2060 SUPER computeCapability: 7.5
coreClock: 1.71GHz coreCount: 34 deviceMemorySize: 7.79GiB deviceMemoryBandwidth: 417.29GiB/s
2020-05-23 15:12:11.607918: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-10.2/lib64:/usr/lib/x86_64-linux-gnu/:
2020-05-23 15:12:11.607931: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-05-23 15:12:11.607939: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-05-23 15:12:11.607946: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-05-23 15:12:11.607954: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-05-23 15:12:11.607961: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-05-23 15:12:11.607969: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-23 15:12:11.607974: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-05-23 15:12:11.607982: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-23 15:12:11.607986: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0 
2020-05-23 15:12:11.607990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N 
```
### Issue: 
- expect `libcudart.so.10.1` can't find
- solution: update `LD_LIBRARY_PATH` with `/usr/local/cuda-10.2/targets/x86_64-linux/lib`: output: same
- solution: make link `libcudart.so.10.1` from `libcudart.so.10.2` in `/usr/local/cuda-10.2/targets/x86_64-linux/lib` ==> SOLVED

### Next issue
- start training and got this error
- in Ipynb:
```
UnknownError:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
	 [[node my_model/conv2d/Conv2D (defined at <ipython-input-5-1e051998210b>:10) ]] [Op:__inference_train_step_568]

Errors may have originated from an input operation.
Input Source operations connected to node my_model/conv2d/Conv2D:
 my_model/Cast (defined at <ipython-input-8-b4778c444eb7>:6)

Function call stack:
train_step
```
- in terminal
```
2020-05-23 19:42:21.077794: E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
```

- [solution:](https://github.com/tensorflow/tensorflow/issues/24496) add following code to define Tesorflow backend ==> SOLVED
```python
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True         # to log device placement
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
```

### Issue: 
- run lesson2 of mdai-lesson with the tensorflow backend fix above - got this error
```
tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
```
- solution: it works when training without jupyter. Temporary solution is with tensorflow - it is better to train outside jupyter.
