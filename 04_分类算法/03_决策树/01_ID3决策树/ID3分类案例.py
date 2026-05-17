"""
ID3决策树 - 隐形眼镜分类案例
运行此脚本前，请确保 trees.py 和 treePlotter.py 在同一目录下
"""
import os
import trees
import treePlotter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_SCRIPT_DIR, "lenses.txt")) as fr:
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearrate']
lensesTree = trees.createTree(lenses, lensesLabels)
print(lensesTree)
print()
print("Done!")
treePlotter.createPlot(lensesTree)
