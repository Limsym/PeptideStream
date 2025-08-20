import os
print("当前工作目录：", os.getcwd())
print("脚本所在目录：", os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import os

# 指定模型文件路径
models_dir = '/models/test.txt'


# 要写的文件路径
test_file_path = models_dir

# 写文件
with open("../models/test.txt", 'w') as f:
    f.write('这是一个测试文件。\n')

print(f'文件已保存到: {test_file_path}')
