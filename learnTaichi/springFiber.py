import taichi as ti
import numpy as np
import taichi.math as tm

# 初始化Taichi
ti.init(arch=ti.cpu)

# 定义质点数量
n = 128

# 定义网格大小
quad_size = 1.0 / n

# 初始化质点坐标矩阵，拥有n*n个3*1质点坐标
pos = ti.Vector.field(3, dtype=float, shape=(n, n))

# 初始化质点速度矩阵，拥有n*n个3*1质点速度
velo = ti.Vector.field(3, dtype=float, shape=(n, n))


# 定义球体


# 质点初始化
@ti.kernel
def mass_point_init():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in pos:
        pos[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        velo[i, j] = [0, 0, 0]


