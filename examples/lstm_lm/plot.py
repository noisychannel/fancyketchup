import sys
import numpy
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    raise Exception('Atleast one numpy archive file is required')

arks = []
labels = []

for i in range(1, len(sys.argv)):
    np_ark = numpy.load(sys.argv[i])
    arks.append(np_ark['history_errs'])
    labels.append(sys.argv[i])

for i in range(len(arks)):
    history_errs = arks[i]
    plt.plot(history_errs[:,0], label=labels[i] + ': valid')
    plt.plot(history_errs[:,1], label=labels[i] + ': test')

plt.ylabel('Average per-word xent')
plt.legend()
plt.show()
