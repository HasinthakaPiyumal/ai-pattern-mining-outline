# Cluster 4

def compile_saliency_function(model):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = model.layers[0].input
    print('-----------------------input-----------------------------')
    print(inp)
    outp = model.layers[-1].output
    print('-----------------------output----------------------------')
    print(outp)
    max_outp = K.T.max(outp, axis=1)
    print(max_outp)
    saliency = K.gradients(K.sum(max_outp), inp)
    print(saliency)
    max_class = K.T.argmax(outp, axis=1)
    print(max_class)
    return K.function([inp, K.learning_phase()], [saliency, max_class])

