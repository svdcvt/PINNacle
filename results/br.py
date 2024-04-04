import os
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

a = '../runs/02.29-11.06.11-subset8-breed_stan'
b = '../runs/02.29-06.30.41-subset8-breed_oob'
pdedirs = [f'{x}-0' for x in range(8)]
name = 'breed_oob_count.csv'


pdf = matplotlib.backends.backend_pdf.PdfPages('breed.pdf')
for pdeid in range(8):
    f = 0
    path = os.path.join(a, pdedirs[pdeid], name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            l = []
            mean, count, total = [], [], []
            for line in f.readlines():
                l.append(list(map(int,line.split(','))))
                total.append(sum(l[-1]))
                count.append(len(l[-1]))
                mean.append(total[-1] / count[-1] if count[-1] else 0)
    else:
        f += 1
        print(path, 'does not exist')
    path = os.path.join(b, pdedirs[pdeid], name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            lb = []
            meanb, countb, totalb = [], [], []
            for line in f.readlines():
                lb.append(list(map(int,line.split(','))))
                totalb.append(sum(lb[-1]))
                countb.append(len(lb[-1]))
                meanb.append(totalb[-1] / countb[-1] if countb[-1] else 0)
    else:
        f += 1
        print(path, 'does not exist')
    if f != 2:
        fig = plt.figure(figsize=(10, 3))
        plt.suptitle(pdedirs[pdeid])
        plt.subplot(131)
        plt.title('mean')
        plt.plot(mean, label='stan')
        plt.plot(meanb, label='oob')
        plt.subplot(132)
        plt.title('count')
        plt.plot(count, label='stan')
        plt.plot(countb, label='oob')
        plt.subplot(133)
        plt.title('total')
        plt.plot(total, label='stan')
        plt.plot(totalb, label='oob')
        plt.legend()
        pdf.savefig(fig)
    else: 
        print(pdedirs[pdeid], 'does not exist')
pdf.close()
