class Optimizers:
    gd = 'gd'
    mgd = 'mgd'
    nag = 'nag'
    rmsprop = 'rmsprop'
    adagrad = 'adagrad'
    adam = 'adam'


class Activations:
    identity = 'identity'
    sigmoid = 'sigmoid'
    relu = 'relu'
    tanh = 'tanh'
    softmax = 'softmax'


class Losses:
    squarederror = 'squarederror'
    crossentropy = 'crossentropy'


class Defaults:
    epochs = 200
    batch_size = 256
    optimizer = Optimizers.adam
    loss = Losses.crossentropy
    learning_rate = 0.001
    momentum_factor = 0.1
    beta1 = 0.9  # 1st moment decay factor
    beta2 = 0.999  # 2nd moment decay factor
    eps = 1e-8