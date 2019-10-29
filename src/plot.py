import pandas as pd
import terminalplot as tp


def plot_data(s, e, path="../data/test.txt"):
    data = pd.read_csv(path, sep="\n", header=None)[0].values[s:e]
    tp.plot(list(data), [x for x in range(e - s)])


def check_zeros():
    plot_data(1060, 1600)


def check_fast():
    plot_data(3679, 4800)


check_zeros()
check_fast()
plot_data(0, 1000, path="../data/train.txt")
