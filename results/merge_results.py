import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('substring', help='substring to include')
parser.add_argument('nosubstring', help='substring to exclude, set `none` is nothing to exclude')
parser.add_argument('tag', help='name for the subdirectory')
parser.add_argument('-i', '--root-directory', default='../runs')
parser.add_argument('-o', '--where', default='./')
args = parser.parse_args()

substring = args.substring
nosubstring = args.nosubstring if args.nosubstring != 'none' else None
root_directory = args.root_directory
tag = args.tag
where = args.where

tagpath = os.path.join(where, tag)
if not os.path.exists(tagpath):
    os.makedirs(tagpath)

errors_cols = '# epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)'.split(', ')

def process_csv(root_dir):
    dfs = []  
    for subdir in os.listdir(root_dir):
        if substring not in subdir or (nosubstring is not None and nosubstring in subdir):
            continue
        csv_path = os.path.join(root_dir, subdir, 'result.csv')
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, usecols=['pde', 'run_time', 'train_loss', 'mse', 'mxe', 'l2rel'])
            df['method'] = subdir.split('-')[-1]
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['pde', 'method']).mean()
    return df

def process_err(root_dir, err='l2res'):
    err_col_id = errors_cols.index(err)
    method_dirs = sorted([subdir for subdir in os.listdir(root_dir) 
                          if substring in subdir and (nosubstring is None or nosubstring not in subdir)],
                         key=lambda x: x.split('-')[-1])
    print("Directories to process:\n", '\n '.join(method_dirs))

    pde_id_name = []
    for mdir in method_dirs:
        # search for dir with biggest number processed pde (just in case)
        csv_path = os.path.join(root_dir, mdir, 'result.csv')
        if os.path.isfile(csv_path):
            id_pde = pd.read_csv(csv_path, usecols=[0,1])
            if len(id_pde) > len(pde_id_name):
                pde_id_name = id_pde
    P = len(pde_id_name)
    M = len(method_dirs)

    pde_ids = [f'{i}-0' for i in range(P)] # for now do not support mean+std
    pde_names = pde_id_name['pde'].values
    
    result_kw_arrays = {
            'methods': [x.split('-')[-1] for x in method_dirs]
            }
    for i, (pde_id, pde_name) in enumerate(zip(pde_ids, pde_names)):
        # 20 : 7 x 20 array : num_pde X num_method X  num_log_iters -> processed further into 20 plots
        pde_errs = []
        for j, mdir in enumerate(method_dirs):
            path = os.path.join(root_dir, mdir, pde_id, 'errors.txt') 
            if os.path.isfile(path):
                pde_errs.append(np.loadtxt(path, usecols=[err_col_id])) # 20
            else:
                pde_errs.append(np.full(1, None))
        l = max(len(x) for x in pde_errs)
        pde_errs = [x if len(x) == l else np.append(x, np.full(l-len(x), None)) for x in pde_errs]
        result_kw_arrays[pde_name] = np.stack(pde_errs, axis=0) # 7 x 20
        print(f'For {pde_name} values size: {result_kw_arrays[pde_name].shape}')
    return result_kw_arrays

def process_loss(root_dir):
    method_dirs = sorted([subdir for subdir in os.listdir(root_dir) 
                          if substring in subdir and (nosubstring is None or nosubstring not in subdir)],
                         key=lambda x: x.split('-')[-1])
    print("Directories to process:\n", '\n '.join(method_dirs))

    pde_id_name = []
    for mdir in method_dirs:
        # search for dir with biggest number processed pde (just in case)
        csv_path = os.path.join(root_dir, mdir, 'result.csv')
        if os.path.isfile(csv_path):
            id_pde = pd.read_csv(csv_path, usecols=[0,1])
            if len(id_pde) > len(pde_id_name):
                pde_id_name = id_pde
    P = len(pde_id_name)
    M = len(method_dirs)

    pde_ids = [f'{i}-0' for i in range(P)] # for now do not support mean+std
    pde_names = pde_id_name['pde'].values
    
    result_kw_arrays = {
            'methods': [x.split('-')[-1] for x in method_dirs]
            }
    for i, (pde_id, pde_name) in enumerate(zip(pde_ids, pde_names)):
        # 20 : 7 x 20 array : num_pde X num_method X  num_log_iters -> processed further into 20 plots
        pde_loss = []
        for j, mdir in enumerate(method_dirs):
            path = os.path.join(root_dir, mdir, pde_id, 'loss.txt') 
            if os.path.isfile(path):
                arr = np.loadtxt(path)
                numloss = (arr.shape[1] - 1) // 3
                weights = arr[:, 1+2*numloss:1+3*numloss]
                train_loss_sum = np.sum(arr[:, 1:1+numloss]/weights, axis=1)
                test_loss_sum = np.sum(arr[:, 1+numloss:1+2*numloss]/weights, axis=1)
                pde_loss.append(train_loss_sum)
                pde_loss.append(test_loss_sum)
            else:
                pde_loss.append(np.full(1, None))
        l = max(len(x) for x in pde_loss)
        pde_loss = [x if len(x) == l else np.append(x, np.full(l-len(x), None)) for x in pde_loss]
        result_kw_arrays[pde_name] = np.stack(pde_loss, axis=0).reshape(M, 2, -1) # 7 x 2 x 20
        print(f'For {pde_name} values size: {result_kw_arrays[pde_name].shape}')
    return result_kw_arrays

result_df = process_csv(root_directory)
result_df.to_csv(os.path.join(tagpath, f'grouped_results-{tag}.csv'))

result_error_kw_arrays = process_err(root_directory)
np.savez(os.path.join(tagpath, f'grouped_errors-{tag}.npz'), **result_error_kw_arrays)

result_loss_kw_arrays = process_loss(root_directory)
np.savez(os.path.join(tagpath, f'grouped_loss-{tag}.npz'), **result_loss_kw_arrays)
