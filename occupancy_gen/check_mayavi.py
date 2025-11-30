from mayavi import mlab
import numpy as np

mlab.options.offscreen = True  # 关键：启用无窗口模式

# 生成测试数据（sin 3D 曲面）
x, y = np.mgrid[-2:2:100j, -2:2:100j]
z = np.sin(x ** 2 + y ** 2)

# 使用 Mayavi 进行 3D 渲染
mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))  # 设定窗口大小和背景颜色
mlab.surf(x, y, z, colormap="coolwarm")  # 绘制 3D 曲面
mlab.view(azimuth=45, elevation=30)  # 设置视角


# 保存渲染结果为图片
mlab.savefig("mayavi_test.png", size=(800, 600))
print("渲染完成，图片已保存为 mayavi_test.png")
