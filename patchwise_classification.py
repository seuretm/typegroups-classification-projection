import argparse
import torch
from torchvision import transforms
from torchvision import utils
from torchvision.datasets import ImageFolder
from network.typegroups_classifier import TypegroupsClassifier

parser = argparse.ArgumentParser()
parser.add_argument("classifier", help="path to the classifier (.tgc file) to use", type=str)
parser.add_argument("count", help="number of patches to extract from each image", type=int)
parser.add_argument("folder", help="path to the folder containing the datasets (subfolders with images)", type=str)
args = parser.parse_args()

sampling = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
inv = transforms.ToPILImage()

test = ImageFolder(args.folder, transform=sampling)


dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tgc = TypegroupsClassifier.load(args.classifier)
test.target_transform = tgc.classMap.get_target_transform(test.class_to_idx)

nb_classes = 1+max(tgc.classMap.cl2id.values())
nb_outputs = tgc.network.fc.out_features
print(nb_classes, 'classes to consider')
print(nb_outputs, 'outputs to process')


sm = torch.nn.Softmax()

nb_good = 0
nb_bad = 0
tgc.network.eval()
imgn = 0
with torch.no_grad():
    f = open("foo.html", "w")
    f.write('<html><head></head><body><table>')
    for sample_num in range(0, test.__len__()):
        score = torch.zeros(1, nb_outputs).to(dev)
        
        patchdict = dict()
        
        f.write('<tr>')
        for n in range(0, args.count):
            sample, target = test.__getitem__(sample_num)
            out, _, _ = tgc.network(sample.unsqueeze_(0).to(dev))
            
            mx, p = torch.max(out, 1)
            
            norm = sm(out)
            conf = torch.max(norm).item()
            
            if mx not in patchdict:
                patchdict[mx] = list()
            
            patchdict[mx].append((norm, out, sample))
            
        nb_kept = 0
        for mx in sorted(patchdict, reverse=True):
            for norm, out, sample in patchdict[mx]:
                
                score += out
                _, p = torch.max(norm, 1)
                
                
                print('>>', out.size())
                res = list()
                for i in tgc.classMap.id2cl:
                    res.append((out.data[0][i], i))
                res = sorted(res, reverse=True)
                
                imgn += 1
                utils.save_image(sample, './img/%d.jpg' % imgn)
                
                f.write('<td>')
                if p.item()==target:
                    f.write('<img src="./img/%d.jpg" border="5" style="border-color:blue"/>' % imgn)
                else:
                    f.write('<img src="./img/%d.jpg" border="5" style="border-color:red"/>' % imgn)
                for s, i in res:
                    f.write('<br> %s: %.2f' % (tgc.classMap.id2cl[i], s))
                
                f.write('</td>\n')
                nb_kept += 1
            if nb_kept >=5:
                break
        print(nb_kept, ' samples kept')
        
        _, p = torch.max(score, 1)
        if p.item()==target:
            f.write('<td>Good (%s)</td>' % tgc.classMap.id2cl[target])
        else:
            f.write('<td>Bad (got %s, wanted %s)</td>' % (tgc.classMap.id2cl[p.item()], tgc.classMap.id2cl[target]))
        f.write('<br/>\n')
        _, p = torch.max(score, 1)
        p = p.item()
        
        if p==target:
            nb_good += 1
        else:
            nb_bad += 1
        print(n, ": ", p, "|", target, "(", score, ")", nb_good,">")
        
        f.write('</tr>')
        
        if nb_good + nb_bad > 15:
            break
            # early exit for test purpose
    f.write('</table></body></htnl>')
    f.close()
print("result: ", nb_good, " good, ", nb_bad, " bad, ", (100.0 * nb_good) / (nb_good + nb_bad), "%")
