# 外部摄像头流网关边界

本项目核心只消费标准媒体，不处理厂商账号。小米接入作为外部适配器示例，不包含私有桥接实现或任何家庭部署记录。

## 责任分离

```text
厂商私有取流 → 独立 access-unit 适配 → loopback RTSP router
                                             → 通用 MediaSource
```

取流网关只负责明确授权的摄像头会话、指定通道、媒体转发和重连，不负责检测、录像策略、模型、巡检、通知或 NAS。

Home Assistant 不在媒体链路内；它可以独立实现可选 `SignalSource`。硬件信号与视频流必须明确绑定，不凭厂商名称猜测来源。

## 已审查来源与许可证

参考来源为 [XiaoMi/xiaomi-miloco](https://github.com/XiaoMi/xiaomi-miloco)，固定审查基线 `1495fac7a4dadb4abbb051ff8663b602e226a093`。

该项目使用自定义 [Xiaomi Miloco License Agreement](https://github.com/XiaoMi/xiaomi-miloco/blob/1495fac7a4dadb4abbb051ff8663b602e226a093/LICENSE.md)，不能当作 Apache/MIT 依赖处理。本仓库不复制或分发其 Python/native 实现、账号会话或私有修改；真正的厂商适配器分发需要另行解决许可兼容性。

本项目只保留通用 `AnnexBAccessUnit`、publisher/consumer 契约和 loopback 安全限制。更多边界见 [第三方许可证](../../THIRD_PARTY_LICENSES.md)。

## 媒体契约

- 原始 H.264/H.265 access unit 须保留 codec、源时间、序号和通道。
- 运行时发现合法通道数，不硬编码设备型号、DID 或某个家庭的通道。
- codec 变化须重建 publisher，不能混写。
- 注册/注销必须成对释放，断流使用有界退避。
- JPEG 截图不是连续媒体覆盖的替代品。
- 源时钟、生成 PTS 与真实覆盖分别记录；合成 read-back 成功不证明真机可靠。

MediaMTX 是可替换外部 router，已有 MIT 来源记录，不捆绑二进制。默认只监听 IP loopback，关闭录制、hooks 和非必要协议。

## 安全与验收

- RTSP URI 无凭据、无查询参数；核心不接触厂商 token。
- 厂商运行树与项目源码隔离，不默认启动额外产品功能、后台任务或云端调用。
- 先合成媒体回读，再有界单路真机验证，最后按明确授权测试多路与恢复。
- 指标包括实际接流、延迟、断流缺口、资源、恢复与证据时间轴。
- 不自动修改 WSL 网络、防火墙或路由器；网络调整必须另行审查。
- 不因环境检查、代码发布或重新打开会话而恢复服务。

配置原则见 [部署边界](../deployment/home-wsl-xiaomi-stream.md)。具体外部桥接命令以其已审查版本为准；核心模板不代表该桥接已经包含或可公开分发。
