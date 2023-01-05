import taichi as ti
import numpy as np
import time
import argparse

# from taichi.misc.util import warning

ti.init(arch=ti.gpu)
width, height = 640, 360

border_temperature = 1000
substeps = 25  # for fast convergence
record_length = 60  # about 2.5s

temperature = ti.field(dtype=ti.f32, shape=(width, height))
img = ti.field(dtype=ti.f32, shape=(width, height, 3))


@ti.kernel
def initialize():
    for i in range(0, height):
        temperature[0, i] = border_temperature
        temperature[width - 1, i] = border_temperature
    for i in range(0, width):
        temperature[i, 0] = border_temperature
        temperature[i, height - 1] = border_temperature


@ti.kernel
def render():
    for i in range(0, width):
        for j in range(0, height):
            img[i, j, 0] = (temperature[i, j]) / border_temperature
            img[i, j, 1] = 0.0
            img[i, j, 2] = (border_temperature - temperature[i, j]) / border_temperature
            # img[i, j, 0] = (temperature[i, j]) / border_temperature
            # img[i, j, 1] = (temperature[i, j]) / border_temperature
            # img[i, j, 2] = (temperature[i, j]) / border_temperature


@ti.kernel
def update():
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            temperature[i, j] = (temperature[i + 1, j] + temperature[i - 1, j] + temperature[i, j - 1] + temperature[
                i, j + 1] + temperature[i, j]) / 5

# 如果当前文件被作为main文件直接运行，则执行后续步骤
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive Ray Tracing')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()

    gui = None
    video_manager = None
    print(f"Is recording: {args.record:b}")
    if not args.record:
        gui = ti.GUI("Hello world", (width, height), show_gui=True)
    else:
        gui = ti.GUI("Hello world", (width, height), show_gui=False)
        video_manager = ti.VideoManager(output_dir="./data", framerate=24, automatic_build=False)
        substeps = 1000
    initialize()

    cnt = 0
    while gui.running:
        for _ in range(substeps):
            update()
        render()
        if not args.record:
            gui.set_image(img)
            gui.show()
        else:
            video_manager.write_frame(img)
            cnt += 1
            if cnt > record_length:
                break

    if args.record:
        print('Exporting .gif')
        video_manager.make_video(gif=True, mp4=False)
        # print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
        print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')