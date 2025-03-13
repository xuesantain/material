import numpy as np
from tmm_fast import coh_tmm as tmm
import tmm_fast.gym_multilayerthinfilm as mltf
from pathlib import Path
import matplotlib.pyplot as plt

path = str(Path(__file__).parent)
pathW = path + '\\tmm_fast\\materials\\nW.txt'
pathGe = path + '\\tmm_fast\\materials\\nGe.txt'
pathSiO2 = path + '\\tmm_fast\\materials\\nSiO2-2.txt'

material_path_list = [pathW, pathGe, pathSiO2]

wl = np.linspace(380, 14000, 3000) * (10**(-9))

N = mltf.get_N(material_path_list, wl.min() * 1e9, wl.max() * 1e9, points=len(wl), complex_n=True)

print(N.shape)
num_layers = 7
num_stacks = 1


a=np.array([0])
print(a)
M = np.ones((num_stacks, num_layers, wl.shape[0]), dtype=np.complex128)

M[:, 0, :] = 1.0 + 0j
M[:, 1, :] = N[2, :]
M[:, 2, :] = N[0, :]
M[:, 3, :] = N[2, :]
M[:, 4, :] = N[1, :]
M[:, 5, :] = N[0, :]
M[:, 6, :] = 1.0 + 0j
# num_layers = 5
# num_stacks = 1
# M = np.ones((num_stacks, num_layers, wl.shape[0]), dtype=np.complex128)


# M[:, 0, :] = N[2, :]
# M[:, 1, :] = N[0, :]
# M[:, 2, :] = N[2, :]
# M[:, 3, :] = N[1, :]
# M[:, 4, :] = N[0, :]


# print(M)
# T = np.array([[np.inf, 85e-9, 5e-9, 71e-9, 500e-9, 100e-9, np.inf]])
# T = np.array([[np.inf,
#         2.156e-07,
#         3.6e-09,
#         2.887e-07,
#         8.35e-08,
#         3.99e-08,
#         np.inf
#     ]])
T=np.array([[np.inf,1.54e-7,5.e-9,2.12e-7,4.96e-7,2.11e-8,np.inf]])
# T = np.array([[np.inf, 85e-9, 5e-9, 71e-9, 500e-9, 100e-9, np.inf]])

# T = np.array([[85e-9, 5e-9, 71e-9, 500e-9, 100e-9]])
O = tmm('s', M, T, wl, lambda_vacuum=wl, device='cpu')

# print(O)
print("--------------------------------")
# print(O['R'])
reflectivity = O['R']
target_array = np.zeros_like(reflectivity)  # 初始化为0
print(target_array.shape)
        # 设置不同波长范围的目标反射率---需要修改
target_array[0, :, (wl >= 0.38e-6) & (wl <= 0.8e-6)] = 0.0  # 0.38-0.8μm 反射率为0
target_array[0, :, (wl >= 3e-6) & (wl <= 5e-6)] = 1.0      # 3-5μm 反射率为1
target_array[0, :, (wl >= 5e-6) & (wl <= 8e-6)] = 0.0      # 5-8μm 反射率为0
target_array[0, :, (wl >= 8e-6) & (wl <= 14e-6)] = 1.0     # 8-14μm 反射率为1
print(target_array)
print(reflectivity.shape)
print(reflectivity[0, 0, :])

mse = np.mean((reflectivity[0, 0, :] - target_array[0, 0, :])**2)
print(f'MSE (Mean Squared Error): {mse:.6f}')



plt.figure(figsize=(10, 6))
plt.plot(wl * 1e6, reflectivity[0, 0, :], label='Reflectivity', color='b')
plt.plot(wl * 1e6, target_array[0, 0, :], label='target', color='r')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Reflectivity')
plt.title('Reflectivity vs Wavelength')
plt.legend()
plt.grid(True)
plt.show()
