from server_side import federated_learning
from tools.load_options import load_config

if __name__ == '__main__':
    """
    Please load the toml configuration you need to experiment with
    example:
    MNIST_LeNet_Lab_Config = load_config(lab_name="(MNIST, LeNet)")
    federated_learning_server_side(MNIST_LeNet_Lab_Config)
    After a while, you will get the calculated results and model
    if you want to change some parameters, you can:
    MNIST_LeNet_Lab_Config['aggregate_function'] = 'layer_defense'
    MNIST_LeNet_Lab_Config['frac'] = 0.02
    then, you will run the federated learning like this:
    federated_learning(MNIST_LeNet_Lab_Config)
    We have three ready-made experimental configurations:
    (CIFAR-10, MobileNet)
    (MNIST, LeNet)
    (CIFAR-100, ResNet-18)
    """
    user_input = input("请输入实验的编号：\n"
                       "1表示MNIST\n"
                       "2表示CIFAR10\n")
    if user_input == "1":
        MNIST_LeNet_Lab_Config = load_config(lab_name="(MNIST, LeNet)")
        federated_learning(MNIST_LeNet_Lab_Config)
    elif user_input == "2":
        CIFAR10_MobileNet_Lab_Config = load_config(lab_name="(CIFAR-10, MobileNet)")
        federated_learning(CIFAR10_MobileNet_Lab_Config)
    else:
        print("编号输入错误！")
