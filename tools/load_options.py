import toml


# 读取配置文件
def load_config(lab_name: "(MNIST, LeNet)"):
    if lab_name == "(MNIST, LeNet)":
        config_file_path = 'configs/mnist_lenet_v5_lab_config.toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    elif lab_name == "(CIFAR-10, MobileNet)":
        config_file_path = 'configs/cifar10_mobilenet_v2_lab_config.toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    elif lab_name == "(CIFAR-100, ResNet-18)":
        config_file_path = 'configs/cifar100_ResNet18_lab_config.toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    elif lab_name == "(IMAGENET, ResNet-18)":
        config_file_path = 'configs/imagenet_resnet18_lab_config.toml'
        with open(config_file_path, 'r') as file:
            config = toml.load(file)
    else:
        raise SystemExit("The Lab name is error!")
    return config['args']
