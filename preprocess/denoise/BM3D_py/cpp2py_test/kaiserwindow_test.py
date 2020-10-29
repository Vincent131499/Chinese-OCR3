import numpy as np


def get_kaiserWindow(kHW):
    kaiserWindow = np.zeros(kHW * kHW)
    if kHW == 8:
        kaiserWindow[0 + kHW * 0] = 0.1924
        kaiserWindow[0 + kHW * 1] = 0.2989
        kaiserWindow[0 + kHW * 2] = 0.3846
        kaiserWindow[0 + kHW * 3] = 0.4325
        kaiserWindow[1 + kHW * 0] = 0.2989
        kaiserWindow[1 + kHW * 1] = 0.4642
        kaiserWindow[1 + kHW * 2] = 0.5974
        kaiserWindow[1 + kHW * 3] = 0.6717
        kaiserWindow[2 + kHW * 0] = 0.3846
        kaiserWindow[2 + kHW * 1] = 0.5974
        kaiserWindow[2 + kHW * 2] = 0.7688
        kaiserWindow[2 + kHW * 3] = 0.8644
        kaiserWindow[3 + kHW * 0] = 0.4325
        kaiserWindow[3 + kHW * 1] = 0.6717
        kaiserWindow[3 + kHW * 2] = 0.8644
        kaiserWindow[3 + kHW * 3] = 0.9718

        for i in range(kHW // 2):
            for j in range(kHW // 2, kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)]

        for i in range(kHW // 2, kHW):
            for j in range(kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j]

    elif kHW == 12:
        kaiserWindow[0 + kHW * 0] = 0.1924
        kaiserWindow[0 + kHW * 1] = 0.2615
        kaiserWindow[0 + kHW * 2] = 0.3251
        kaiserWindow[0 + kHW * 3] = 0.3782
        kaiserWindow[0 + kHW * 4] = 0.4163
        kaiserWindow[0 + kHW * 5] = 0.4362
        kaiserWindow[1 + kHW * 0] = 0.2615
        kaiserWindow[1 + kHW * 1] = 0.3554
        kaiserWindow[1 + kHW * 2] = 0.4419
        kaiserWindow[1 + kHW * 3] = 0.5139
        kaiserWindow[1 + kHW * 4] = 0.5657
        kaiserWindow[1 + kHW * 5] = 0.5927
        kaiserWindow[2 + kHW * 0] = 0.3251
        kaiserWindow[2 + kHW * 1] = 0.4419
        kaiserWindow[2 + kHW * 2] = 0.5494
        kaiserWindow[2 + kHW * 3] = 0.6390
        kaiserWindow[2 + kHW * 4] = 0.7033
        kaiserWindow[2 + kHW * 5] = 0.7369
        kaiserWindow[3 + kHW * 0] = 0.3782
        kaiserWindow[3 + kHW * 1] = 0.5139
        kaiserWindow[3 + kHW * 2] = 0.6390
        kaiserWindow[3 + kHW * 3] = 0.7433
        kaiserWindow[3 + kHW * 4] = 0.8181
        kaiserWindow[3 + kHW * 5] = 0.8572
        kaiserWindow[4 + kHW * 0] = 0.4163
        kaiserWindow[4 + kHW * 1] = 0.5657
        kaiserWindow[4 + kHW * 2] = 0.7033
        kaiserWindow[4 + kHW * 3] = 0.8181
        kaiserWindow[4 + kHW * 4] = 0.9005
        kaiserWindow[4 + kHW * 5] = 0.9435
        kaiserWindow[5 + kHW * 0] = 0.4362
        kaiserWindow[5 + kHW * 1] = 0.5927
        kaiserWindow[5 + kHW * 2] = 0.7369
        kaiserWindow[5 + kHW * 3] = 0.8572
        kaiserWindow[5 + kHW * 4] = 0.9435
        kaiserWindow[5 + kHW * 5] = 0.9885

        for i in range(kHW // 2):
            for j in range(kHW // 2, kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)]

        for i in range(kHW // 2, kHW):
            for j in range(kHW):
                kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j]

    else:
        for k in range(kHW * kHW):
            kaiserWindow[k] = 1.0

    kaiserWindow = kaiserWindow.reshape((kHW, kHW))
    return kaiserWindow


def get_kaiserWindow_np(kHW):
    k = np.kaiser(kHW, 2)
    k_2d = k[:, np.newaxis] @ k[np.newaxis, :]
    return k_2d


if __name__ == '__main__':
    import numpy as np

    kaiser = get_kaiserWindow(8)
    kaiser_np = get_kaiserWindow_np(8)
    diff = np.abs(kaiser-kaiser_np)
    print(np.max(diff))
    print(np.sum(diff))
