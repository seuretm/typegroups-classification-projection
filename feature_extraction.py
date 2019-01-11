import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy
from random import shuffle
from network.typegroups_classifier import TypegroupsClassifier

parser = argparse.ArgumentParser()
parser.add_argument("classifier", help="path to the classifier (.tgc file) to use", type=str)
parser.add_argument("count", help="number of patches to extract from each image", type=int)
parser.add_argument("folder", help="path to the folder containing the datasets (subfolders with images)", type=str)
parser.add_argument("output", help="path to the file to store result data", type=str)
args = parser.parse_args()

sampling = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

inv = transforms.ToPILImage()

test = ImageFolder(args.folder, transform=sampling)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

tgc = TypegroupsClassifier.load(args.classifier)
target_transform = tgc.classMap.get_target_transform(test.class_to_idx)


# confusion matrix
cm = [[0 for x in range(tgc.network.fc.out_features)] for y in range(tgc.network.fc.out_features)] 


nb_good = 0
nb_bad = 0
tgc.network.eval()
imgn = 0


feature_file  = open(args.output, 'wb')

with torch.no_grad():
    idx = [i for i in range(0, test.__len__())]
    shuffle(idx)
    for sample_num in idx:
        sample, target = test.__getitem__(sample_num)
        print(target)
        
        score = torch.zeros(1, tgc.network.fc.out_features).to(dev)
        features = torch.zeros(1, tgc.network.fc.in_features).to(dev)
        print(features.size())
        for n in range(0, args.count):
            sample, target = test.__getitem__(sample_num)
            out, _, ap = tgc.network(sample.unsqueeze_(0).to(dev))
            ap = ap.view(ap.size(0), -1)
            score += out
            features += ap
            feature_file.write(target.to_bytes(1, byteorder='big', signed=True))
            numpy.save(feature_file, ap.cpu().data.numpy())
        _, p = torch.max(score, 1)
        
        print('target:', target,'\tres:', p.item())
        if target_transform(target)>=0:
            cm[target_transform(target)][p.item()] += 1            
            print(tgc.classMap.id2cl[target])
        print(cm)

feature_file.close()
