def f1(precision, recall):
    if precision + recall  == 0:
        return 0
    return round(2 * precision * recall / (precision + recall), 4)




