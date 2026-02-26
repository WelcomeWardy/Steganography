def lr_schedule(epoch):

    if epoch < 200:
        return 0.001
    elif epoch < 400:
        return 0.0003
    elif epoch < 600:
        return 0.0001
    else:
        return 0.00003