"""
支持向量机 - 手写数字识别案例
运行此脚本前，请确保 SVM算法.py 在同一目录下
使用 sklearn 内置数据集，无需外部文件
"""
import SVM算法 as svmMLiA

if __name__ == '__main__':
    svmMLiA.testDigitsSklearn(('rbf', 20))
    print()
    print("Done!")
