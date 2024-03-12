from tools.view_img import view_image_cifar10, view_image_mnist


def mark_a_sig_logo_mnist(image):
    for i in range(1, 7):
        image[:, :, i * 4] = 0.5
    return image, 5


def mark_a_sig_logo_cifar10(image):
    for i in range(1, 8):
        image[:, :, i * 4] = 255
    return image, 5


def poison_data_with_sig(image, dataset_name):
    if dataset_name == "cifar10":
        image, label = mark_a_sig_logo_cifar10(image)
        # view_image_cifar10(image, save_path="./imgs/sig_cifar10.png")
    elif dataset_name == "mnist":
        image, label = mark_a_sig_logo_mnist(image)
        # view_image_mnist(image, save_path="./imgs/sig_mnist.png")
    return image
