import os
import pandas as pd
import numpy as np

substring = 'all'
root_directory = '../runs'
tag = '0216'
where = './'

errors_cols = '# epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)'.split(', ')

def process_csv(root_dir):
    dfs = []  
    for subdir in os.listdir(root_dir):
        if substring not in subdir:
            continue
        csv_path = os.path.join(root_dir, subdir, 'result.csv')
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path, usecols=['pde', 'run_time', 'train_loss', 'mse', 'mxe', 'l2rel'])
            df['method'] = subdir.split('_')[-1]
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(['pde', 'method']).mean()
    return df

def process_err(root_dir, err='l2res'):
    err_col_id = errors_cols.index(err)
    method_dirs = [subdir for subdir in os.listdir(root_dir) if substring in subdir]
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
            'methods': [x.split('_')[-1] for x in method_dirs]
            }
    for i, (pde_id, pde_name) in enumerate(zip(pde_ids, pde_names)):
        # 20 : 7 x 20 array : num_pde X num_method X  num_log_iters -> processed further into 20 plots
        pde_errs = []
        for j, mdir in enumerate(method_dirs):
            path = os.path.join(root_dir, mdir, pde_id, 'errors.txt') 
            if os.path.isfile(path):
                pde_errs.append(np.loadtxt(path, usecols=[err_col_id])) # 20
        result_kw_arrays[pde_name] = np.stack(pde_errs, axis=0) # 7 x 20
    return result_kw_arrays


result_df = process_csv(root_directory)
result_df.to_csv(f'{where}/grouped_results-{tag}.csv')

result_error_kw_arrays = process_err(root_directory)
np.savez(f'{where}/grouped_errors-{tag}.npz', **result_error_kw_arrays)
