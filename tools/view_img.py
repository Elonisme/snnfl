import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def view_image_mnist(image, save_path):
    image = image.permute(1, 2, 0)
    img_tensor = torch.clamp(image, 0, 1)
    image_np = (img_tensor.numpy() * 255).astype(np.uint8)  # 将0-1范围映射到0-255，并转为uint8类型
    pil_image = Image.fromarray(image_np.squeeze(), mode='L')  # 'L'表示灰度模式
    plt.imshow(np.squeeze(image_np), cmap='gray')
    pil_image.save(save_path)
    raise SystemExit("Image was successfully marked")


def view_image_cifar10(image, save_path):
    image = image.permute(1, 2, 0)
    img_tensor = torch.clamp(image, 0, 1)
    image_np = (img_tensor.numpy() * 255).astype(np.uint8)  # 将0-1范围映射到0-255，并转为uint8类型
    pil_image = Image.fromarray(image_np.squeeze())  # 'L'表示灰度模式
    plt.imshow(np.squeeze(image_np))
    pil_image.save(save_path)
    raise SystemExit("Image was successfully marked")
