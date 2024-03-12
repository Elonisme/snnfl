from PIL import Image
from torchvision.transforms import ToTensor

from tools.view_img import view_image_mnist, view_image_cifar10


def blended_two_images(image, dataset_name):
    blended_image_path = "./attacks/imgs/ball32.png" if dataset_name == 'cifar10' else "./attacks/imgs/ball28.png"
    blended_image = Image.open(blended_image_path)

    # 将PIL图像转换为torch张量
    to_tensor = ToTensor()
    torch_blended = to_tensor(blended_image).to(image.dtype)

    # 叠加两张图像
    if dataset_name == 'cifar10':
        result_image = image + torch_blended[0:3, :, :]
        # view_image_cifar10(image=result_image, save_path="./imgs/blended_cifar10.png")
        return result_image
    elif dataset_name == 'mnist':
        result_image = image + torch_blended[0, :, :]
        # view_image_mnist(image=result_image, save_path="./imgs/blended_mnist.png")
        return result_image


def poison_data_with_blended(image, dataset_name):
    poisonous_image = blended_two_images(image, dataset_name)
    return poisonous_image
