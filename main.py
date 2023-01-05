import tensorflow as tf
import torch
import taichi as ti
# taichi中间已经引用了numba库，所以这里引用会出现重复引用错误
# from numba import jit,njit


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# for Apple silicon ,try ti.metal
ti.init(arch=ti.cpu)

@ti.kernel
def hello(i:ti.i32):
    a=40
    print('Hello world!',a+i)

# @jit
# def numbaTest(a,b):
#     return a*b

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')


hello(2)

print("Hello Dasiy")

print("tf version",tf.__version__)

# 这条指令被废弃了
print("tf gpu",tf.test.is_gpu_available())
# tf.config.list_physical_devices('GPU')

print("torch version",torch.__version__)

torch.ones(1)+torch.ones(1)

