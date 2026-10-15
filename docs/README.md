# 文档索引

公开文档只描述通用实现、配置与限制。家庭试验日志、机器标识、精确房间映射、媒体、标注与运行输出不作为项目文档发布。

## 项目与架构

- [项目入口](../README.md)
- [产品与安全规范](../SPEC.md)
- [实现状态与开发门槛](../DEVELOPMENT_PLAN.md)
- [技术架构](architecture.md)
- [业务架构](business-architecture.md)
- [变更日志](../CHANGELOG.md)

## 使用与开发

- [CLI 参考](cli-reference.md)与[配置参考](configuration.md)：由生成脚本维护。
- [WSL 原生运行模板](deployment/home-wsl-runtime.md)
- [外部摄像头桥接边界](integrations/xiaomi-stream-gateway.md)
- [小米流适配配置](deployment/home-wsl-xiaomi-stream.md)
- [延迟日报与恢复](deferred-reports.md)
- [可选本地身份登记](identity/local-face-enrollment.md)
- [规则引擎](rules.md)

## 安全与发布

- [威胁模型](security/threat-model.md)
- [第三方许可证](../THIRD_PARTY_LICENSES.md)
- [安全政策](../SECURITY.md)
- [贡献指南](../CONTRIBUTING.md)
- [公开验证约定](verification/README.md)

示例必须使用匿名 ID、占位符或合成数据。发布前审计工作区、staged index 和生成的源码包。历史私人记录仅在本地保留，不因整理公开文档而扩大媒体授权。
