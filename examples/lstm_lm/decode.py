"""
Decoder for the LSTM-LM. Will generate sentences
starting with a few tokens using trained model params
"""

from __future__ import print_function

import os
import sys
import time
import numpy

from cutils.params.utils import zipp, load_params
from cutils.data_interface.utils import pad_and_mask

# Include current path in the pythonpath
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)

from lm import LSTM_LM

SEED = 123
numpy.random.seed(SEED)


def decode_lstm(
    load_from='lstm_model.npz'
):
    npz_archive = numpy.load(load_from)
    model_options = npz_archive['model_options']
    ptb_data = npz_archive['ptb_data']

    lstm_lm = LSTM_LM(model_options['dim_proj'], model_options['y_dim'],
                      ptb_data.dictionary, SEED)

    print('Reloading params from %s' % save_to)
    load_params(load_from, lstm_lm.params)
    # Update the tparams with the new values
    zipp(lstm_lm.params, lstm_lm.tparams)

    print("model options", model_options)

    # Create the shared variables for the model
    lstm_lm.build_decode()
    test_sentences = ['with the', 'the cat', 'when the']
    test_sentences = [ptb_data.dictionary.read_sentence(s) for s in test_sentences]
    test_sentences, test_mask = pad_and_mask(test_sentences)

    start_time = time.time()
    output = lstm_lm.f_decode(test_sentences, test_mask, model_options['maxlen'])
    end_time = time.time()

    print('Decoding took %.1fs' % (end_time - start_time))

if __name__ == '__main__':
    decode_lstm()
