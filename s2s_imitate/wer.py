#-*- coding: utf-8 -*-

import numpy


def wer(ref, hyp, show_result=False):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this:
        wer("This great machine can recognize speech".split(), "This machine can wreak a nice beach".split())
        wer(['1', '2', '3', '4', '5', '6'], ['1', '3', '4', '7', '8', '9', '10'], show_result=True)
        wer([1, 2, 3, 4, 5, 6], [1, 3, 4, 7, 8, 9, 10])
    """

    #build the matrix
    d = numpy.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=numpy.uint8).reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref)+1):
        for j in range(len(hyp)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    result = float(d[len(ref)][len(hyp)]) / len(ref) * 100

    # print the result in aligned way
    if show_result:
        if isinstance(ref[0], int):
            print("If you want to see the WER result, you should input a word not an id.")
        else:
            # find out the manipulation steps
            x = len(ref)
            y = len(hyp)
            li = []
            while True:
                if x == 0 and y == 0:
                    break
                else:
                    if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                        li.append("e")
                        x -= 1
                        y -= 1
                    elif d[x][y] == d[x][y - 1] + 1:
                        li.append("i")
                        x = x
                        y -= 1
                    elif d[x][y] == d[x - 1][y - 1] + 1:
                        li.append("s")
                        x -= 1
                        y -= 1
                    else:
                        li.append("d")
                        x -= 1
                        y = y
            li = li[::-1]

            print("REF:", end=' ')
            for i in range(len(li)):
                if li[i] == "i":
                    count = 0
                    for j in range(i):
                        if li[j] == "d":
                            count += 1
                    index = i - count
                    print(" " * (len(hyp[index])), end=' ')
                elif li[i] == "s":
                    count1 = 0
                    for j in range(i):
                        if li[j] == "i":
                            count1 += 1
                    index1 = i - count1
                    count2 = 0
                    for j in range(i):
                        if li[j] == "d":
                            count2 += 1
                    index2 = i - count2
                    if len(ref[index1])<len(hyp[index2]):
                        print(ref[index1] + " " * (len(hyp[index2]) - len(ref[index1])), end=' ')
                    else:
                        print(ref[index1], end=' ')
                else:
                    count = 0
                    for j in range(i):
                        if li[j] == "i":
                            count += 1
                    index = i - count
                    print(ref[index], end=' ')
            print()
            print("HYP:", end=' ')
            for i in range(len(li)):
                if li[i] == "d":
                    count = 0
                    for j in range(i):
                        if li[j] == "i":
                            count += 1
                    index = i - count
                    print(" " * (len(ref[index])), end=' ')
                elif li[i] == "s":
                    count1 = 0
                    for j in range(i):
                        if li[j] == "i":
                            count1 += 1
                    index1 = i - count1
                    count2 = 0
                    for j in range(i):
                        if li[j] == "d":
                            count2 += 1
                    index2 = i - count2
                    if len(ref[index1]) > len(hyp[index2]):
                        print(hyp[index2] + " " * (len(ref[index1]) - len(hyp[index2])), end=' ')
                    else:
                        print(hyp[index2], end=' ')
                else:
                    count = 0
                    for j in range(i):
                        if li[j] == "d":
                            count += 1
                    index = i - count
                    print(hyp[index], end=' ')
            print()
            print("EVA:", end=' ')
            for i in range(len(li)):
                if li[i] == "d":
                    count = 0
                    for j in range(i):
                        if li[j] == "i":
                            count += 1
                    index = i - count
                    print("D" + " " * (len(ref[index]) - 1), end=' ')
                elif li[i] == "i":
                    count = 0
                    for j in range(i):
                        if li[j] == "d":
                            count += 1
                    index = i - count
                    print("I" + " " * (len(hyp[index]) - 1), end=' ')
                elif li[i] == "s":
                    count1 = 0
                    for j in range(i):
                        if li[j] == "i":
                            count1 += 1
                    index1 = i - count1
                    count2 = 0
                    for j in range(i):
                        if li[j] == "d":
                            count2 += 1
                    index2 = i - count2
                    if len(ref[index1]) > len(hyp[index2]):
                        print("S" + " " * (len(ref[index1]) - 1), end=' ')
                    else:
                        print("S" + " " * (len(hyp[index2]) - 1), end=' ')
                else:
                    count = 0
                    for j in range(i):
                        if li[j] == "i":
                            count += 1
                    index = i - count
                    print(" " * (len(ref[index])), end=' ')
            print()

        result = str("%.2f" % result) + "%"
        print("WER: " + result)
        print()

    return result
