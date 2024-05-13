import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse

from cycler import cycler

parser = argparse.ArgumentParser()
parser.add_argument('tag')
parser.add_argument('--range', nargs=3, default=[0, -1, 1], type=int)
args = parser.parse_args()

name = f'{args.tag}/grouped_results-{args.tag}'
grouped_df = pd.read_csv(name + '.csv', index_col=[0,1], usecols=[0, 1, 2, 6]) # index, runtime, l2rel

all_pdes = grouped_df.index.unique(level='pde')
all_methods =  grouped_df.index.unique(level='method')
M = len(all_methods)
P = len(all_pdes)

cm = matplotlib.cm.get_cmap('gist_ncar')
custom_cycler = lambda n: cycler(color=[cm(x) for x in np.linspace(0, 1, n+1)])

pdf = matplotlib.backends.backend_pdf.PdfPages(name + '.pdf')
# bar-plots of runtime and l2rel (last epoch)
K = grouped_df.shape[1]
RN = min(3, P) # maximum number of plots in a row
axx = lambda a, b: axes[a, b] if RN>1 else axes[a]
for pp in range(0, P, RN):
    fig, axes = plt.subplots(K, RN, figsize=(5*RN + RN, 3 * K + K), sharex=True)
    for k, figname in enumerate(grouped_df.columns):
        for i, pde in enumerate(all_pdes[pp:pp + RN]):
            res = grouped_df.iloc[:, k]
            res = res.loc[pde]
            methods = [m for m in all_methods if m in res.index]
            axx(0,i).set_title(pde)
            axx(k,i).bar(methods, res.loc[methods])
            axx(1,i).set_yscale('log')
            axx(1,i).xaxis.set_tick_params(rotation=90)
        axx(k,0).set_ylabel(f"{'log-' * k}{figname}")                
    pdf.savefig(fig, bbox_inches="tight")    
pdf.close()

name = f'{args.tag}/grouped_errors-{args.tag}'
error = 'l2rel'
results_errors = np.load(name + '.npz', allow_pickle=True)
methods_order = results_errors['methods']
pdes = list(results_errors.keys())[1:]
P = len(pdes)

# line plots of l2rel across iterations for each pde
pdf = matplotlib.backends.backend_pdf.PdfPages(name + '.pdf')
RN = min(3, P) # maximum number of plots in a row
axx = lambda a: axes[a] if RN>1 else axes
for pp in range(0, P, RN):
    fig, axes = plt.subplots(1, RN, figsize=(6 * RN + RN, 5))
    for i, pde in enumerate(pdes[pp: pp + RN]):
        axx(i).set_title(pde)
        res_pde = results_errors[pde]
        axx(i).set_prop_cycle(custom_cycler(len(res_pde)))
        for j in range(len(res_pde)):
            end = len(res_pde[j]) if args.range[1] == -1 else args.range[1]
            xt = np.arange(args.range[0], end + 1, dtype=int)
            axx(i).plot(xt[:-2], res_pde[j][slice(*args.range)], label=methods_order[j])
            print(j, methods_order[j])
            axx(i).set_xticks(xt[::(end-args.range[0]) // 10])
        axx(i).set_yscale('log')
        axx(0).set_ylabel(error)
    axx(i).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pdf.savefig(fig, bbox_inches="tight")
pdf.close()

name = f'{args.tag}/grouped_loss-{args.tag}'
results_loss = np.load(name + '.npz', allow_pickle=True)
methods_order = results_loss['methods']
pdes = list(results_loss.keys())[1:]
P = len(pdes)

# line plots of l2rel across iterations for each pde
pdf = matplotlib.backends.backend_pdf.PdfPages(name + '.pdf')
RN = min(3, P) # maximum number of plots in a row
axx = lambda a: axes[a] if RN>1 else axes
for pp in range(0, P, RN):
    fig, axes = plt.subplots(1, RN, figsize=(6 * RN + RN, 5))
    for i, pde in enumerate(pdes[pp: pp + RN]):
        axx(i).set_title(pde)
        res_pde = results_loss[pde]
        axx(i).set_prop_cycle(custom_cycler(len(res_pde)))
        for j in range(len(res_pde)):
            end = len(res_pde[j][0]) if args.range[1] == -1 else args.range[1]
            xt = np.arange(args.range[0], end + 1, dtype=int)
            p=axx(i).plot(xt[:-2], res_pde[j][0][slice(*args.range)], label=methods_order[j])
            axx(i).plot(xt[:-2], res_pde[j][1][slice(*args.range)], linestyle='--', c=p[0].get_color())
            axx(i).set_xticks(xt[::(end-args.range[0]) // 10])
        axx(i).set_yscale('log')
        axx(0).set_ylabel('full loss')
    axx(i).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pdf.savefig(fig, bbox_inches="tight")
pdf.close()
