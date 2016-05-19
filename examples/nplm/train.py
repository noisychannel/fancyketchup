import os
import sys
import theano
import theano.tensor as T
import numpy
from collections import OrderedDict
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cutils.training.trainer import new_sgd
from cutils.training.utils import get_minibatches_idx
from cutils.numeric import numpy_floatX

# Include current path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

from nplm import NPLM
from setimes import SeTimes


def sgd_optimization_nplm_mlp(learning_rate=1., L1_reg=0.0, L2_reg=0.0001,
                              n_epochs=1000, dataset='../../data/settimes',
                              batch_size=1000, n_in=150, n_h1=750, n_h2=150,
                              context_size=4, use_nce=False, nce_k=100,
                              use_dropout=False, dropout_p=0.5):
    SEED = 1234

    st_data = SeTimes(dataset, emb_dim=n_in)
    print("... Creating the partitions")
    train, valid = st_data.load_data(context_size=context_size)
    print("... Done creating partitions")

    print("... Building the model")
    # Symbolic variables for input and output for a batch
    x = T.imatrix('x')
    y = T.ivector('y')
    lr = T.scalar(name='lr')

    emb_x = st_data.dictionary.Wemb[x.flatten()] \
        .reshape([x.shape[0], context_size * n_in])

    rng = numpy.random.RandomState(SEED)
    trng = RandomStreams(SEED)
    use_noise = theano.shared(numpy_floatX(0.))

    nce_q = st_data.dictionary.noise_distribution
    nce_samples = T.imatrix('noise_s')

    model = NPLM(
        rng=rng,
        input=emb_x,
        n_in=context_size * n_in,
        n_h1=n_h1,
        n_h2=n_h2,
        n_out=st_data.dictionary.num_words(),
        use_nce=use_nce
    )

    tparams = OrderedDict()
    for i, nplm_m in enumerate(model.params):
        tparams['nplm_' + str(i)] = nplm_m
    tparams['Wemb'] = st_data.dictionary.Wemb

    # Cost to minimize
    if use_nce:
        cost = model.loss(y, nce_samples, nce_q)
    else:
        # MLE via softmax
        cost = model.loss(y)
    # Add L2 reg to the cost
    cost += L2_reg * model.L2

    grads = T.grad(cost, wrt=list(tparams.values()))

    if use_nce:
        f_cost = theano.function([x, y, nce_samples], cost, name='f_cost')
        f_grad_shared, f_update = new_sgd(lr, tparams, grads,
                                          cost, x, y, nce_samples)
    else:
        f_cost = theano.function([x, y], cost, name='f_cost')
        f_grad_shared, f_update = new_sgd(lr, tparams, grads,
                                          cost, x, y)

    print("... Optimization")
    kf_valid = get_minibatches_idx(len(valid[0]), batch_size)
    print("%d training examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))

    disp_freq = 10
    valid_freq = len(train[0]) // batch_size
    save_freq = len(train[0]) // batch_size

    uidx = 0
    estop = False
    start_time = time.time()
    for eidx in range(n_epochs):
        n_samples = 0
        # Shuffle and get training stuff
        kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
        for _, train_index in kf:
            uidx += 1
            use_noise.set_value(1.)

            x_batch = [train[0][t] for t in train_index]
            y_batch = [train[1][t] for t in train_index]
            # Convert x and y into numpy objects
            x_batch = numpy.asarray(x_batch, dtype='int32')
            y_batch = numpy.asarray(y_batch, dtype='int32')

            if use_nce:
                # Create noise samples to be passed as well
                # Expected size is (bs, k)
                # Don't sample UNK and PAD
                noisy_samples = numpy.random.randint(
                    2, st_data.dictionary.num_words(),
                    size=(x_batch.shape[0], nce_k), dtype='int32'
                )
                loss = f_grad_shared(x_batch, y_batch, noisy_samples)
            else:
                loss = f_grad_shared(x_batch, y_batch)
            f_update(learning_rate)

            if numpy.isnan(loss) or numpy.isinf(loss):
                print('bad cost detected: ', loss)
                return 1., 1.

            if numpy.mod(uidx, disp_freq) == 0:
                print('Epoch', eidx, 'Update', uidx, 'Cost', loss)

if __name__ == '__main__':
    sgd_optimization_nplm_mlp(dataset=sys.argv[1], use_nce=True)
