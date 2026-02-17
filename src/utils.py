def normalize(x, min_x, max_x):
    if max_x == min_x:
        return 0.0
    return (x - min_x) / (max_x - min_x)
