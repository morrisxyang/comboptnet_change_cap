import numpy as np

if __name__ == '__main__':
    x2 = np.asarray([1, 0, 0, 0, 1, 0, 0, 1, 1, 0])
    price = np.asarray([37.2, 19, 37, 35.4, 35.5, 42.8, 14.5, 19.2, 34.8, 25.5])
    x_opt = np.asarray([0, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    print(x2.dot(price))
    print(x_opt.dot(price))
    print(x_opt - x2)
    print(42.8 + 14.5 -19.2-37.2)

    print((300* 10* (1-0.7864) *5)/300)
    print(300* 10* (1-0.7864))

    weight = [18, 31, 35, 25, 28, 15, 27, 20, 22, 30]
    print(sum(weight))
