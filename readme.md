# Tensorrt 推理工程

这个程序用于测试Tensorrt模型在jetson设备的推理时间



JetPack版本：5.1.1

Tensorrt版本：8.5.2



一.编译命令

```
mkdir build&cd build
cmake ../
make
```

二. 运行程序

```
./trt_infer_app <engine_path>
```

