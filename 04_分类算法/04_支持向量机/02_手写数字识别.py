"""
支持向量机 - 手写数字识别案例
运行此脚本前，请确保 SVM算法.py 在同一目录下
"""
import SVM算法 as svmMLiA

if __name__ == '__main__':
    svmMLiA.testDigits(('rbf', 20))
    print()
    print("Done!")
