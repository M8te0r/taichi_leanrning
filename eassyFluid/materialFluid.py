import taichi as ti
import numpy as np
import taichi.math as tm
import matplotlib.cm as cm

# 初始化Taichi
ti.init(arch=ti.gpu)

'''----------------几何参数设置----------------'''
# 定义场大小
grid_x = 150
grid_y = 600
Q = 9

# 定义圆柱坐标
c_x = grid_x / 2  # 圆柱圆心x坐标
c_y = grid_y / 4  # 圆柱圆心y坐标
c_r = grid_x / 9  # 圆柱半径长度

# 定义D2Q9分量权重
# 1/36 1/9 1/36
# 1/9  4/9 1/9
# 1/36 1/9 1/36
weight_arry = np.array(
    [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0],
    dtype=np.float32)

weight = ti.field(ti.f32, shape=9)
weight.from_numpy(weight_arry)

# 定义D2Q9分量索引
# 6 2 5
# 3 0 1
# 7 4 8
e_arry = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                   [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)

e = ti.field(ti.i32, shape=(9, 2))
e.from_numpy(e_arry)

'''----------------初始条件设置----------------'''
# 定义初始条件
steps = 40000  # 总计算次数
Re = 200.0  # 模型要模拟的雷诺数
u_max = 0.1  # 粒子最大速度，也是粒子初始速度
nu = 2 * u_max * c_r / Re  # 模型流体的运动粘性系数，（这里用的是半径，严格来说应该用直径）
# nu = u_max * 150 / Re
tau = 3 * nu + 0.5  # LBGK(单松弛模型)的弛豫时间
inv_tau = 1 / tau  # 松弛因子提前求逆，降低运算量
rho = ti.field(ti.f32, shape=(grid_x, grid_y))  # 密度场
velo = ti.Vector.field(n=2, dtype=float, shape=(grid_x, grid_y))  # 速度场，由每个速度向量由x和y方向的速度分量表示
pixel = ti.Vector.field(n=3, dtype=float, shape=(grid_x, grid_y))  # 像素场，每个像素由rgb参数表示
pixel_2 = ti.field(ti.f32, shape=(grid_x, grid_y))
mask_flag = ti.field(ti.i32, shape=(grid_x, grid_y))  # 阴影遮蔽flag，用于在渲染时确定障碍物的位置
f_old = ti.Vector.field(n=9, dtype=float, shape=(grid_x, grid_y))  # 旧分布状态
f_new = ti.Vector.field(n=9, dtype=float, shape=(grid_x, grid_y))  # 新分布状态


# 建立边界
@ti.func
def boundary_establish(temp_x: int, temp_y: int):
    # 初始化
    mask_flag[temp_x, temp_y] = 0

    # 外部边界（左、右、下、上）
    if (temp_x == 0 or temp_x == grid_x - 1 or temp_y == 0 or temp_y == grid_y - 1):
        mask_flag[temp_x, temp_y] = 1

    # 内部边界（圆柱）
    if (((ti.cast(temp_x, ti.f32) - c_x) ** 2 + (ti.cast(temp_y, ti.f32) - c_y) ** 2.0) <= c_r ** 2.0):
        mask_flag[temp_x, temp_y] = 2


# 平衡态离散速度方程
# 计算网格(i,j)所表示晶格内，第k个方向上的平衡态的离散速度
@ti.func
def equilibrium(i, j, k):
    eu = ti.cast(e[k, 0], ti.f32) * velo[i, j][0] + ti.cast(e[k, 1], ti.f32) * velo[i, j][1]
    v_2 = velo[i, j][0] ** 2.0 + velo[i, j][1] ** 2.0
    return weight[k] * rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * v_2)


# 初始化
@ti.kernel
def init():
    for i in range(0, grid_x):
        for j in range(0, grid_y):
            boundary_establish(i, j)
            velo[i, j][0] = 0
            velo[i, j][1] = 0
            rho[i, j] = 1.0
            for k in ti.static(range(9)):
                f_new[i, j][k] = equilibrium(i, j, k)
                f_old[i, j][k] = f_new[i, j][k]


# 碰撞扩散阶段
# 更新分布函数状态
# 发生碰撞后，粒子发生完全弹性碰撞到达碰撞后的坐标
# 在碰撞后的坐标下，通过平衡分布函数进一步得到粒子最终处于稳定态的坐标
# 更新(i,j)表示的每个网格中所有分量(k)的状态分布
@ti.kernel
def collide_spread():
    for i, j in ti.ndrange((1, grid_x - 1), (1, grid_y - 1)):
        for k in ti.static(range(9)):
            i_prev = i - e[k, 0]
            j_prev = j - e[k, 1]
            f_new[i, j][k] = f_old[i_prev, j_prev][k] - inv_tau * (
                    f_old[i_prev, j_prev][k] - equilibrium(i_prev, j_prev, k))


# 更新内部网格（排除边界）的物理参数(速度，密度)
@ti.kernel
def update_status():
    for i, j in ti.ndrange((1, grid_x - 1), (1, grid_y - 1)):
        rho[i, j] = 0.0
        velo[i, j][0] = 0.0
        velo[i, j][1] = 0.0
        # 更新(i,j)网格的第k个分量的速度
        for k in ti.static(range(9)):
            f_old[i, j][k] = f_new[i, j][k]
            rho[i, j] += f_new[i, j][k]
            velo[i, j][0] += (ti.cast(e[k, 0], ti.f32) * f_new[i, j][k])
            velo[i, j][1] += (ti.cast(e[k, 1], ti.f32) * f_new[i, j][k])
        velo[i, j][0] /= rho[i, j]
        velo[i, j][1] /= rho[i, j]


# 边界条件的更新方式
@ti.func
def update_boundary(i, j, i_neigbhor, j_neigbhor):
    # 更新边界密度
    rho[i, j] = rho[i_neigbhor, j_neigbhor]

    # 更新分量的物理参数
    for k in ti.static(range(9)):
        f_old[i, j][k] = equilibrium(i, j, k) \
                         - equilibrium(i_neigbhor, j_neigbhor, k) \
                         + f_old[i_neigbhor, j_neigbhor][k]


@ti.kernel
def boundary_handle_2():
    for j in range(grid_y):
        # 左
        velo[0, j][0] = 0.0
        velo[0, j][1] = 0.0
        update_boundary(0, j, 1, j)
        # 右
        velo[grid_x - 1, j][0] = 0.0
        velo[grid_x - 1, j][1] = 0.0
        update_boundary(grid_x - 1, j, grid_x - 2, j)
    for i in range(grid_x):
        # 下
        velo[i, 0][0] = 0.0
        velo[i, 0][1] = 0.1
        update_boundary(i, 0, i, 1)
        # 上
        velo[i, grid_y - 1][0] = velo[i, grid_y - 2][0]
        velo[i, grid_y - 1][1] = velo[i, grid_y - 2][1]
        update_boundary(i, grid_y - 1, i, grid_y - 2)
    for i, j in ti.ndrange(grid_x, grid_y):
        # 内边界处理
        i_neigbhor = 0
        j_neigbhor = 0
        if (mask_flag[i, j] == 2):
            if (i >= c_x):
                i_neigbhor = i + 1
            else:
                i_neigbhor = i - 1
            if (j >= c_y):
                j_neigbhor = j + 1
            else:
                j_neigbhor = j - 1

            velo[i, j][0] = 0.0
            velo[i, j][1] = 0.0
            update_boundary(i, j, i_neigbhor, j_neigbhor)


# 根据流速情况设置像素
# FIXME v可能要除以0.15
@ti.kernel
def set_pixel():
    for i, j in ti.ndrange(grid_x, grid_y):
        # v = ti.sqrt(velo[i, j][0] ** 2 + velo[i, j][1] ** 2)
        v = velo[i, j][1]
        pixel[i, j].xyz = v / u_max, 0.0, 1 - v / u_max
        pixel_2[i, j] = velo[i, j][1]
        # pixel_2[i, j] = rho[i, j]


''' --------------------------------------------------------------------------------------------------- '''


# 渲染
def render():
    gui = ti.GUI('lbm solver', (grid_x, grid_y))
    init()
    for step in range(steps):
        if (step % 1000 == 0):
            print('Step: {:}'.format(step))
        collide_spread()
        update_status()
        boundary_handle_2()
        set_pixel()
        gui.set_image(cm.plasma(pixel_2.to_numpy() / 0.15))
        gui.show()


render()
