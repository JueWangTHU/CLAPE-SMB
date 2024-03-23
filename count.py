# -*- coding: utf-8 -*-
# @Time         : 2024/3/23 10:53
# @Author       : Jue Wang and Yufan Liu
# @Description  : Counting the number of positive and negative samples

def count(path):
    f = open(path, mode='r')
    lines = f.readlines()
    zero = one = 0
    for line in lines:
        line = line.strip()
        if line.startswith('0') or line.startswith('1'):
            for c in line:
                if c == '0' :
                    zero += 1
                else:
                    one += 1
    List = [zero, one]
    return List

if __name__ == "__main__":
    List = count("./example.txt")
    print(List)
    print(List[1] / (List[1] + List[0]))
