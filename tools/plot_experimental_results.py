from torch.utils.tensorboard import SummaryWriter


def initialize_summary_writer():
    return SummaryWriter()


def plot_line_chart(writer, accuracy, type_str, step):
    # 将准确值和类型字符串写入Tensorboard
    writer.add_scalar(f'{type_str}', accuracy, global_step=step)
