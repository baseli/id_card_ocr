### 身份证复印件识别离线

#### 环境
1. `python 3.7.3`
2. `python -m venv venv`
#### 运行
1. `pip install -r requirements.txt`
2. 修改 `ocr.py` 中的文件路径部分
3. `python ocr.py` 等待输出

#### 计划
- [x] 引入 `paddleocr`
- [x] 框选身份证位置，透视变化，进行识别
- [ ] 格式化结果
- [ ] `windows` 编译 `exe` 执行
- [ ] `windows gui` 编写
