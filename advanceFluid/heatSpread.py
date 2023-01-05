from numpy import *; from numpy.linalg import *

import matplotlib.pyplot as plt

#使用numba对循环进行加速
from numba import jit,njit;

import pandas as pd
model_max = pd.DataFrame(columns = ['u_max','y_max','v_max','x_max'])

maxIter = 200000 #最大循环次数
nx = 128; ny = 128;q = 9   #网格数目和离散速度个数
##########MRT-D2Q9速度计算模型#################################
c = array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
           [1,1],[-1,1],[-1,-1],[1,-1]]) #离散速度坐标
t = 1./36. * ones(q)    #权系数
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.
i1 = array([3,6,7])  #右边界未知速度
i3 = array([1,8,5])  #左边界未知速度
i4 = array([4,7,8])  #上边界未知速度
i5 = array([2,5,6])  #下边界未知速度
sumpop = lambda fin: sum(fin,axis=0) #密度计算求和函数
#定义变换矩阵M
M = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1]])
Minv = inv(M)  #M的逆矩阵
nulb = 0.1   #定义运动粘度
omega = 1./(3*nulb+0.5)  #松弛参数
print(omega)
S = diag([0,1.90,1.63,0,1.14,0,1.14,omega,omega])  #松弛矩阵
MinvS = matmul(Minv, S)
#定义平衡态计算函数
def equilibrium(rho,u):
    cu   = 3.0 * dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = zeros((q,nx,ny))
    for i in range(q): feq[i,:,:] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq
#定义MRT-平衡态计算函数
def MRT_equilibrium(rho,x):
    u = x[0]; v = x[1]
    a0 = rho; a1 = multiply(rho,(-2.+3.*(u**2+v**2)))
    a2 = multiply(rho,(1. - 3.*(u**2+v**2)))
    a3 = multiply(rho,u)
    a4 = -multiply(rho,u)
    a5 = multiply(rho,v)
    a6 = -multiply(rho,v)
    a7 = multiply(rho,(u**2-v**2))
    a8 = multiply(rho,(multiply(u,v)))
    return array([a0,a1,a2,a3,a4,a5,a6,a7,a8])
#定义计算MRT离散力的格式
def MRT_force(u,T):
    f = zeros((2,nx,ny))
    f[1] = -gbei*(T-0.5)
    f0 = zeros((nx,ny));f1 = 6.*(1.-omega/2.)*(f[0]*u[0]+f[1]*u[1])
    f2 = -6.*(1.-omega/2.)*(f[0]*u[0]+f[1]*u[1])
    f3 = f[0] ; f4 = -(1.-omega/2.)*f[0]
    f5 = -f[1] ; f6 = -(1.-omega/2.)*f[1]
    f7 = 2.*(1.-omega/2.)*(f[0]*u[0]-f[1]*u[1])
    f8 = (1.-omega/2.)*(f[0]*u[1]+f[1]*u[0])
    return array([f0,f1,f2,f3,f4,f5,f6,f7,f8])
###############MRT-D2Q5温度计算模型##########################################
qt = 5  #温度离散速度个数
nulb_d = nulb/0.71   #热扩散系数，0.71表示空气普朗特数
omega_g = 1./(5*nulb_d/2+0.5)  #热松弛参数
S_t = diag([1,omega_g,omega_g,omega_g,omega_g])  #热松弛矩阵
#变换矩阵
M_t = array([[1,1,1,1,1],
            [0,1,0,-1,0],
            [0,0,1,0,-1],
            [0,1,-1,1,-1],
            [-4,1,1,1,1],
            ])
#热离散速度
ct = array([[0,0],[1,0],[0,1],[-1,0],[0,-1]])
#权重系数
t_t = array([0.2,0.2,0.2,0.2,0.2])
#热平衡态计算函数
@njit
def tem_equilibrium(T,u):
    feq = zeros((qt,nx,ny))
    for i in range(nx):
        for j in range(ny):
            for kt in range(qt):
                feq[kt,i,j] = t_t[kt]*T[i,j]*(1.+3.*(ct[kt,0]*u[0,i,j]+ct[kt,1]*u[1,i,j]))
    return feq
Q = inv(M_t)
#初始化速度
rho = ones((nx,ny))
u = zeros((2,nx,ny))
T0 = full((nx,ny),0.5);T0[0,:] = 1.0;T0[-1,:] = 0.0;T=T0.copy()

R = [1000, 10000, 1.e5, 1.e6]  # 瑞利数变化范围
for Ra in R:
    gbei = Ra * nulb * nulb_d / ny ** 3
    fin = equilibrium(rho, u);
    gin = tem_equilibrium(T, u)
    F = full((2, nx, ny), gbei)
    F[0] = 0.0
    error0 = 1.0;
    time = 0

    while time < maxIter and error0 > 1.e-8:

        # 速度矩空间碰撞过程
        m = tensordot(M, fin, (1, 0))
        m_f_eq = MRT_equilibrium(rho, u)
        m_force_eq = MRT_force(u, T)
        m_0 = tensordot(S, (m - m_f_eq), (1, 0))  # 计算碰撞的一部分
        m_out = m - m_0 + m_force_eq  # 碰撞
        fout = tensordot(Minv, m_out, (1, 0))  # 从矩空间转化到速度空间

        # 温度矩空间碰撞
        geq = tem_equilibrium(T, u)
        geq = tensordot(M_t, geq, (1, 0))
        gm = tensordot(M_t, gin, (1, 0))
        gou = gm - tensordot(S_t, (gm - geq), (1, 0))
        gout = tensordot(Q, gou, (1, 0))

        # 迁移过程
        for i in range(q):
            fin[i, :, :] = roll(roll(fout[i, :, :], c[i, 0], axis=0), c[i, 1], axis=1)
        for i in range(qt):
            gin[i, :, :] = roll(roll(gout[i, :, :], ct[i, 0], axis=0), ct[i, 1], axis=1)

        # 上边界处理
        fin[i4, :, -1] = fin[i5, :, -1]
        gin[:, :, -1] = gin[:, :, -2]
        # 下边界处理
        fin[i5, :, 0] = fin[i4, :, 0]
        gin[:, :, 0] = gin[:, :, 1]
        # 左边界处理
        fin[i3, 0, :] = fin[i1, 0, :]
        gin[1, 0, :] = 0.4 * T0[0, :] - gin[2, 0, :]
        # 右边界处理
        fin[i1, -1, :] = fin[i3, -1, :]
        gin[3, -1, :] = 0.4 * T0[-1, :] - gin[4, -1, :]
        # 计算宏观变量
        rho = sumpop(fin)
        u0 = u.copy()
        F[1] = -gbei * (T - 0.5)
        u = (dot(c.transpose(), fin.transpose((1, 0, 2))) + F / 2.) / rho
        T = sumpop(gin)
        # 计算收敛程度
        temp1 = sqrt((u[0, :, :] - u0[0, :, :]) ** 2 + (u[1, :, :] - u0[1, :, :]) ** 2).sum()
        temp2 = sqrt(u[0, :, :] ** 2 + u[1, :, :] ** 2).sum()
        error0 = temp1 / (temp2 + 1.e-30)
        # print(error0)
        time += 1
    plt.clf();
    plt.figure(figsize=(5, 5))
    plt.contourf(sqrt(u[0] ** 2 + u[1] ** 2).transpose())
    plt.savefig("./work/velocity_" + str(Ra) + ".png")  # 百度work目录下可保存运行文件
    plt.axis('off')
    plt.show()
    # 绘制温度分布
    plt.clf();
    plt.figure(figsize=(5, 5))
    plt.contourf(T.transpose())
    plt.savefig("./work/temperature_" + str(Ra) + ".png")
    plt.axis('off')
    plt.show()
    # 绘制流线图
    Y, X = mgrid[0:nx, 0:ny]
    plt.clf();
    plt.figure(figsize=(5, 5))
    plt.streamplot(X, Y, u[1], u[0])
    plt.savefig("./work/stream_" + str(Ra) + ".png")
    plt.axis('off')
    plt.show()
    model_max.loc['Ra' + str(Ra)] = array([u[0].max(), argmax(u[0]), u[1].max(), argmax(u[1])])