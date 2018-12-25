# tensorflow-in-action

## 配置环境

### 设置 pip 源

```
// 打开相关的配置文件
vim ~/.pip/pip.conf

// 写入以下内容
 [global]
 trusted-host=mirrors.aliyun.com
 index-url=http://mirrors.aliyun.com/pypi/simple

// 保存退出
```

### 创建虚拟运行环境

```
// 如果没安装 virtualenv 的话，先安装 virtualenv
sudo pip install virtualenv

// 切换到项目根目录下，执行以下命令
virtualenv --no-site-packages --python=python3.5 venv

// 进入虚拟环境
source venv/bin/activate

// 安装各种依赖包
pip install numpy pandas matplotlib tensorlfow==1.0.0

// 退出虚拟环境
deactivate
```

## 脚本介绍

- gpu-test.py: 测试是否有GPU

