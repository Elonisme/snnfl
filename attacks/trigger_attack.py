from tools.view_img import view_image_mnist, view_image_cifar10


def mark_a_two_times_two_white_dot(image):
    image[:, :1, -1:] = 255
    return image


def mark_a_five_pixel_white_plus_logo(image):
    image[:, 0, -2] = 255
    image[:, 1, -3:] = 255
    image[:, 2, -2] = 255
    return image


def poison_data_with_trigger(image, dataset_name):
    if dataset_name == 'mnist':
        image = mark_a_two_times_two_white_dot(image)
        # view_image_mnist(image, save_path="./imgs/trigger_mnist.png")
        return image, 0
    elif dataset_name == 'cifar10':
        image = mark_a_five_pixel_white_plus_logo(image)
        # view_image_cifar10(image, save_path="./imgs/trigger_cifar10.png")
        return image, 0
    else:
        print(dataset_name)
        raise ValueError('Unknown dataset')
