from __future__ import print_function, division
import argparse
import os
import matplotlib.pyplot as plt
import numpy
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("folder", help="path to the folder containing the datasets (subfolders with images)", type=str)
args = parser.parse_args()


bf = open('test.dat', 'rb')

labels = list()
features = list()
while True:
    try:
        lbl = int.from_bytes(bf.read(1), byteorder='big', signed=True)
        arr = numpy.load(bf)
        labels.append(lbl)
        features.append(arr[0])
    except:
        break
bf.close()

print(len(features), ' data points loaded')


test = ImageFolder(args.folder)

perplexity = 10
rate=1000

#TODO: replace
plot_lbl = test.classes

tsne = TSNE(n_components=2, perplexity=perplexity, verbose=3, n_iter=300, learning_rate=rate).fit_transform(features)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
for l in range(len(plot_lbl)):
    idx = [i for i, x in enumerate(labels) if x==l]
    if len(idx)>0:
        ax1.scatter(tsne[idx[:],0], tsne[idx[:],1], s=64, marker='.', label=plot_lbl[l])
#plt.xlim(-300, 300)
#plt.ylim(-300, 300)
plt.legend(loc='upper left');
plt.savefig('tsne-features.png', bbox_inches='tight')

quit()


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
#for l in range(8):
for l in (0, 1, 6, 7):
    idx = [i for i, x in enumerate(labels) if x==l]
    print(idx)
    print(labels)
    print(len(tsne[idx,0]))
    ax1.scatter(tsne[idx[:300],0], tsne[idx[:300],1], s=64, marker='.', label=plot_lbl[l])
#plt.xlim(-300, 300)
#plt.ylim(-300, 300)
plt.legend(loc='upper left');
plt.savefig('tsne-features-anchors-ul.png', bbox_inches='tight')

plt.legend(loc='upper right');
plt.savefig('tsne-features-anchors-ur.png', bbox_inches='tight')

plt.legend(loc='lower right');
plt.savefig('tsne-features-anchors-lr.png', bbox_inches='tight')

plt.legend(loc='lower left');
plt.savefig('tsne-features-anchors-ll.png', bbox_inches='tight')

xl = plt.xlim()
yl = plt.ylim()

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
for l in (2, 3, 4, 5):
    idx = [i for i, x in enumerate(labels) if x==l]
    print(idx)
    print(labels)
    print(len(tsne[idx,0]))
    ax1.scatter(tsne[idx[:300],0], tsne[idx[:300],1], s=64, marker='.', label=plot_lbl[l])
#plt.xlim(-300, 300)
#plt.ylim(-300, 300)
plt.xlim(xl)
plt.ylim(yl)
plt.legend(loc='upper left');
plt.savefig('tsne-features-others-ul.png', bbox_inches='tight')

plt.legend(loc='upper right');
plt.savefig('tsne-features-others-ur.png', bbox_inches='tight')

plt.legend(loc='lower right');
plt.savefig('tsne-features-others-lr.png', bbox_inches='tight')

plt.legend(loc='lower left');
plt.savefig('tsne-features-others-ll.png', bbox_inches='tight')

quit()


tsne = TSNE(n_components=2, perplexity=perplexity, verbose=3, n_iter=10000, learning_rate=rate).fit_transform(scores)
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
for l in range(8):
#for l in (0, 1, 6, 7):
    idx = [i for i, x in enumerate(labels) if x==l]
    ax1.scatter(tsne[idx[:300],0], tsne[idx[:300],1], s=64, marker='.', label=plot_lbl[l])
#plt.xlim(-300, 300)
#plt.ylim(-300, 300)
plt.legend(loc='upper left');
plt.savefig('tsne-scores.png', bbox_inches='tight')
