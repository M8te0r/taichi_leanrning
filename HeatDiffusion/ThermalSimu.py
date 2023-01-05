import taichi as ti
import numpy as np
import taichi.math as tm

# 参考自 https://github.com/bjmiao/taichi_hw1_diffusion

# 初始化Taichi
ti.init(arch=ti.cpu)

# 定义渲染窗体属性
width, height = 640, 360

# 定义初始温度
inner_temperture = 100

# 定义一个2D温度场
temperature = ti.field(dtype=ti.f32, shape=(width, height))

# 定义子步长
substeps = 25

# 定义一个帧（即一张图）
img = ti.field(dtype=ti.f32, shape=(width, height, 3))


def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(["0{0:x}".format(v) if v < 16 else
                          "{0:x}".format(v) for v in RGB])


# 初始化场内每个区域的温度
@ti.kernel
def initialize():
    # 因为taichi会对最外层的for循环强制并行执行，所以没有使用嵌套for写法
    for i in range(0, height):
        temperature[0, i] = inner_temperture
        temperature[width - 1, i] = inner_temperture
    for i in range(0, width):
        temperature[i, 0] = inner_temperture
        temperature[i, height - 1] = inner_temperture


# FIXME 定义最终渲染函数(打算以2d的方式渲染，但是内部计算是基于3d计算的)
# 颜色过渡算法 https://bsouthga.dev/posts/color-gradients-with-python
# img的第三个量使用来表示RGB颜色的
@ti.kernel
def render():
    for i in range(0, width):
        for j in range(0, height):
            img[i, j, 0] = (temperature[i, j]) / inner_temperture
            img[i, j, 1] = 0.0
            img[i, j, 2] = (inner_temperture - temperature[i, j]) / inner_temperture


# 场内温度更新方式，取每个网格上下左右中心5个元素取平均
@ti.kernel
def update_1():
    for i in range(1, width - 1):
        for j in range(1, height - 1):
            temperature[i, j] = (temperature[i + 1, j] + temperature[i - 1, j] + temperature[i, j - 1] + temperature[
                i, j + 1] + temperature[i, j]) / 5


@ti.kernel
def update_2(start_x: int, start_y: int):
    for i in range(start_x, width - 1):
        for j in range(start_y, height - 1):
            temperature[i, j] = (temperature[i + 1, j] + temperature[i - 1, j] + temperature[i, j - 1] + temperature[
                i, j + 1] + temperature[i, j]) / 5


# 定义gui界面
gui = ti.GUI("heat diffusal", (width, height), show_gui=True)

# 初始化
initialize()

# 循环渲染
while gui.running:
    for _ in range(substeps):
        update_2(1, 1)
    render()
    gui.set_image(img)
    gui.show()
