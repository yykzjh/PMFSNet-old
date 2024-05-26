# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/22 22:12
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import numpy as np
import pandas as pd

from proplot import rc
import matplotlib.pyplot as plt



def generate_bubble_image():
    def circle_area_func(x, p=75, k=150):
        return np.where(x < p, (np.sqrt(x / p) * p) * k, x * k)

    def inverse_circle_area_func(x, p=75, k=150):
        return np.where(x < p * k, (((x / k) / p) ** 2) * p, x / k)

    rc["font.family"] = "Times New Roman"
    rc["axes.labelsize"] = 36
    rc["tick.labelsize"] = 32
    rc["suptitle.size"] = 28
    rc["title.size"] = 28

    data = pd.read_excel(r"./experience_data.xlsx", sheet_name="data01")
    model_names = data.Method
    FLOPs = data.FLOPs
    Params = data.Params
    values = data.IoU
    xtext_positions = [2250, 20, 2300, 1050, 225, 1010, 200, 600, 1810, 360, 975, 10]
    ytext_positions = [68, 80.5, 54, 73, 70, 82, 83.5, 84, 74, 78, 86.5, 86]
    legend_sizes = [1, 5, 25, 50, 100, 150]
    legend_yposition = 57.5
    legend_xpositions = [590, 820, 1030, 1260, 1480, 1710]
    p = 15
    k = 150

    # 画图
    fig, ax = plt.subplots(figsize=(18, 9), dpi=100, facecolor="w")
    pubble = ax.scatter(x=FLOPs, y=values, s=circle_area_func(Params, p=p, k=k), c=list(range(len(model_names))), cmap=plt.cm.get_cmap("Spectral"), lw=3, ec="white", vmin=0, vmax=11)
    center = ax.scatter(x=FLOPs[:-1], y=values[:-1], s=20, c="#e6e6e6")
    ours_ = ax.scatter(x=FLOPs[-1:], y=values[-1:], s=60, marker="*", c="red")

    # 添加文字
    for i in range(len(FLOPs)):
        ax.annotate(model_names[i], xy=(FLOPs[i], values[i]), xytext=(xtext_positions[i], ytext_positions[i]), fontsize=32, fontweight=(200 if i < (len(FLOPs) - 1) else 600))
    for i, legend_size in enumerate(legend_sizes):
        ax.text(legend_xpositions[i], legend_yposition, str(legend_size) + "M", fontsize=32, fontweight=200)

    # 添加图例
    kw = dict(prop="sizes", num=legend_sizes, color="#e6e6e6", fmt="{x:.0f}", linewidth=None, markeredgewidth=3, markeredgecolor="white", func=lambda s: np.ceil(inverse_circle_area_func(s, p=p, k=k)))
    legend = ax.legend(*pubble.legend_elements(**kw), bbox_to_anchor=(0.7, 0.15), title="Parameters (Params) / M", ncol=6, fontsize=0, title_fontsize=0, handletextpad=90, frameon=False)

    ax.set(xlim=(0, 2900), ylim=(45, 90), xticks=np.arange(0, 2900, step=300), yticks=np.arange(45, 90, step=5), xlabel="Floating-point Operations Per Second (FLOPs) / GFLOPs",
           ylabel="Intersection over Union (IoU) / %")

    fig.tight_layout()
    fig.savefig("./3D_CBCT_Tooth_bubble_image.jpg", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    # 生成气泡图
    generate_bubble_image()
