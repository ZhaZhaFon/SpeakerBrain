# SpkBrain README文档

## 文件组织

以VoxCeleb1+x-vector为例说明代码仓库的文件组织架构

/speakerbrain: 根目录
---/README.md: 说明文档
---/recipe: 不同数据集的代码仓库
------/VoxCeleb1: 数据集名称
---------/SpeakerRecognition: 说话人识别任务
------------/xvector: x-vector算法
---------------/train.py
---------------/Makefile
---/speakerbrain: 通用代码仓库
------/audio.py: 用于处理音频


