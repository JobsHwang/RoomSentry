# 第三方许可证边界

项目自有代码与文档采用 Apache-2.0。该许可证不覆盖依赖、权重、数据、媒体、摄像头桥接或外部运行时；能下载不等于可商用或可再分发。

## 分发范围

公开源码只包含项目实现、合成测试和来源/配置示例。不包含模型权重、第三方二进制、家庭媒体、私人标注、运行数据库或厂商私有运行树。安装核心不会自动下载它们。

下表保留已有固定版本的许可记录，不替代原许可证或法律审查。升级、组合部署和再分发须重新核对代码、运行时、权重和训练数据的独立条款。

## 依赖与工具

| 组件 | 边界 |
|---|---|
| Python 标准库 | 核心运行不要求视觉或厂商依赖 |
| OpenCV 4.12.0.88、Ultralytics 8.4.115 | 可选 vision extra；遵守各自许可证，不捆绑源码或 wheel |
| Ultralytics 实现及其 ByteTrack/BoT-SORT | 按 Ultralytics 的 AGPL-3.0/商业许可边界处理，不能套用算法原仓库的 MIT |
| cryptography 49.0.0 及传递依赖 | 可选身份加密 runtime；以精确安装包的 LICENSE/NOTICE 为准 |
| FFmpeg/ffprobe | 外部工具，适用具体构建的 LGPL/GPL 与第三方条款 |
| MediaMTX | 外部可替换 RTSP router；已有来源记录为 MIT，不分发其二进制 |
| PyTorch、Transformers、CUDA 和模型专用包 | 独立隔离运行时；CUDA 仍受 NVIDIA 条款约束 |
| Ruff、setuptools、GitHub 官方 actions | 开发/构建依赖，不由本仓库重新许可；CI action 固定完整 commit SHA |

## 可选模型来源

| 模型/来源 | 固定来源与限制 |
|---|---|
| YOLO11 detector / Pose | [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11)；AGPL-3.0 来源记录，含 pickle checkpoint 风险；精确元数据见 config 中 reviewed manifest 示例 |
| YOLOE-26m | [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)；只作独立实验，不是默认依赖或获准生产模型 |
| MobileCLIP2 编码器 | [Apple 模型许可](https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS)；研究范围限制独立存在，不能由 Ultralytics AGPL 声明推导产品使用权限 |
| MiniCPM-V-4.6 | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)，revision 8169864629825dc1d755a5aa1cd8b5935dcbc83f；模型卡 Apache-2.0 记录，不捆绑权重 |
| Mage-VL | [microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL)，revision 5c78cab61938e73859b63724d9bf5cb88c477eaa；模型卡 Apache-2.0 记录；remote code 必须独立审查和固定 |
| SmolVLM2-2.2B-Instruct | [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)，revision 482adb537c021c86670beed01cd58990d01e72e4；Apache-2.0 模型卡记录 |
| Bonsai 2 27B | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)，revision b072e1d3b35a0a630cece372c2127528e0994386；该 revision 的 Apache-2.0 记录不覆盖运行时或 CUDA |
| Prism llama.cpp runtime | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)，commit adfffbe41b2cabcd51fff326ab045662265062bb；MIT 代码记录，发布二进制与动态库仍独立核对 |
| OpenCV Zoo YuNet / SFace | [OpenCV Zoo](https://github.com/opencv/opencv_zoo)；YuNet 目录 MIT、SFace 目录 Apache-2.0 的固定审查记录；不能推导其他人脸权重或训练集同样获准 |
| DMHead 专用头部模型 | [PINTO0309/DMHead](https://github.com/PINTO0309/DMHead)；仓库 MIT、模型目录 GPLv3、CrowdHuman 数据非商用研究/教育条款分开处理，不进入发行物 |

没有任何来源登记自动授权下载、执行 remote code、读取家庭媒体或上线。模型来源、大小、哈希和 manifest 必须在实际准入时复核；未知权重不能冒充已审查版本。

## 参考实现与数据

- ByteTrack、OC-SORT 与 Deep SORT 的论文/原仓库只提供机制参考。项目自有 motion matcher 不声称兼容或复刻这些实现。
- Frigate、Double Take、Viseron 等仅作适配器和策略研究；未复制的实现不构成本项目依赖，第三方识别权重仍须单独审计。
- CAVIAR 的公开资料要求 CC BY-SA 与项目致谢，具体 CC 版本未明确；原 XML、视频与派生输出不随源码分发。
- Pexels、NASA 等外部媒体即便可以研究使用，也需核对肖像、隐私、授权与再分发条件，不自动进入 CI 或源码包。
- 婴幼儿、人脸、睡眠和哭声数据集不因论文公开而获得训练、下载或分发许可。gated、非商用、许可不明或涉及未成年人隐私的数据默认不纳入。
- 私人媒体与标注始终不是开源数据集；不公开素材哈希、逐帧模型响应或家庭实验轨迹。
- Mage 的可选 codec-video-prep 路径因许可证/来源及依赖一致性问题未准入分发，不添加为默认依赖。

## 摄像头与厂商桥接

- [XiaoMi/xiaomi-miloco](https://github.com/XiaoMi/xiaomi-miloco) 使用自定义 Xiaomi Miloco License Agreement，不按 OSI 开源许可处理。本仓库不复制、打包或重新许可该项目、native library 或私有修改运行树。
- 其他厂商桥接在确认准确上游 URL、许可证和再分发条件前，不纳入安装流程。
- Home Assistant 适配器使用公开事件契约编写项目自有规范化代码，不代表 Home Assistant 或任意 SDK 被重新许可。

## 发布门槛

1. 核查新增组件精确版本、LICENSE/NOTICE、模型权重和数据来源。
2. 不将外部依赖、媒体或模型缓存打入源码包。
3. 运行隐私审计，保留必要归属声明与可复现 source SBOM。
4. source SBOM 只反映源码与声明依赖，不证明实际部署已安装、无漏洞或取得生产使用许可。
5. 存在不明确条款时停止相关再分发；不要据项目 Apache-2.0 作许可推断。

本文件是工程边界记录，不构成法律意见。
