def func(N, M, c, h):
    res = 0
    map_dictionary  = [(time_to_cook, hunger_satisfied) for time_to_cook, hunger_satisfied in zip(c, h)]
    map_dictionary = sorted(map_dictionary, key=lambda x: x[1], reverse=True)

    print(map_dictionary)

    for time, hunger in map_dictionary:
        if N == 0 or M == 0:
            return res

        if time <= M:
            res += hunger
            N -= 1
            M -= 1

    return res
