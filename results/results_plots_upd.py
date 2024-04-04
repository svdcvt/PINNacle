import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tag')
args = parser.parse_args()

name = f'{args.tag}/grouped_results-{args.tag}'
grouped_df = pd.read_csv(name + '.csv', index_col=[0,1], usecols=[0, 1, 2, 6]) # index, runtime, l2rel

all_pdes = grouped_df.index.unique(level='pde')
all_methods =  grouped_df.index.unique(level='method')
M = len(all_methods)
P = len(all_pdes)

pdf = matplotlib.backends.backend_pdf.PdfPages(name + '.pdf')
# bar-plots of runtime and l2rel (last epoch)
K = grouped_df.shape[1]
RN = 4 # maximum number of plots in a row
for pp in range(0, P, RN):
    fig, axes = plt.subplots(K, RN, figsize=(5*RN + RN, 3 * K + K), sharex=True)
    for k, figname in enumerate(grouped_df.columns):
        for i, pde in enumerate(all_pdes[pp:pp + RN]):
            res = grouped_df.iloc[:, k]
            res = res.loc[pde]
            methods = [m for m in all_methods if m in res.index]
            axes[0,i].set_title(pde)
            axes[k,i].bar(methods, res.loc[methods])
            axes[1,i].set_yscale('log')
            axes[k,0].set_ylabel(f"{'log-' * k}{figname}")
    pdf.savefig(fig)    
pdf.close()

name = f'{args.tag}/grouped_errors-{args.tag}'
error = 'l2rel'
results_errors = np.load(name + '.npz', allow_pickle=True)
methods_order = results_errors['methods']
pdes = list(results_errors.keys())[1:]
P = len(pdes)

# line plots of l2rel across iterations for each pde
pdf = matplotlib.backends.backend_pdf.PdfPages(name + '.pdf')
RN = 4 # maximum number of plots in a row
for pp in range(0, P, RN):
    fig, axes = plt.subplots(1, RN, figsize=(5 * RN + RN, 5))
    for i, pde in enumerate(pdes[pp: pp + RN]):
        axes[i].set_title(pde)
        res_pde = results_errors[pde]
        for j in range(len(res_pde)):
            axes[i].plot(res_pde[j], label=methods_order[j])
        axes[i].set_yscale('log')
        axes[0].set_ylabel(error)
    axes[i].legend()
    pdf.savefig(fig)
pdf.close()

