import os
import argparse
import pickle
import torch
import logging

# from utils.datasets import PackedConformationDataset
from evaluation.covmat import CovMatEvaluator, print_covmat_results
# from utils.misc import *
# from utils import get_model, get_optimizer, get_scheduler, get_logger, get_new_log_dir
def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def results2latex(path, df_cov, df_mat):
    if 'qm9' in path.lower():
        cov = df_cov.loc[0.5]
    elif 'drugs' in path.lower():
        # name=1.2500000000000002
        # cov = df_cov.iloc[24]
        cov = df_cov.loc[1.2500000000000002]
    cov_r_mean = f"{cov['COV-R_mean']:.4f}"
    cov_r_median = f"{cov['COV-R_median']:.4f}"
    cov_p_mean = f"{cov['COV-P_mean']:.4f}"
    cov_p_median = f"{cov['COV-P_median']:.4f}"

    mat = df_mat.iloc[0]
    mat_r_mean = f"{mat['MAT-R_mean']:.4f}"
    mat_r_median = f"{mat['MAT-R_median']:.4f}"
    mat_p_mean = f"{mat['MAT-P_mean']:.4f}"
    mat_p_median = f"{mat['MAT-P_median']:.4f}"
    with open(path, 'w+') as f:
        f.writelines(f'{cov_r_mean} {cov_r_median} {mat_r_mean} {mat_r_median} {cov_p_mean} {cov_p_median} {mat_p_mean} {mat_p_median}\n')
        latex_line = 'name & \multicolumn{1}{c|}{'+ cov_r_mean +'}&'+cov_r_median +\
              '&\multicolumn{1}{c|}{'+mat_r_mean+'}&'+ mat_r_median +\
                  '& \multicolumn{1}{c|}{' + cov_p_mean + '} &' + cov_p_median +\
                    '& \multicolumn{1}{c|}{'+ mat_p_mean+'} &' + mat_p_median +'\\\ \hline'
        f.writelines(latex_line)
    print(f'Save latex text in {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=2)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--use_force_field', action='store_true', default=False)
    args = parser.parse_args()
    assert os.path.isfile(args.path)

    if args.use_force_field:
        tag = args.path.split('/')[-1].split('.')[0] + '_useFF'
    else:
        # Logging
        tag = args.path.split('/')[-1].split('.')[0]
    logger = get_logger('eval', os.path.dirname(args.path), 'log_eval_'+ tag +'.txt')
    
    # Load results
    logger.info('Loading results: %s' % args.path)
    with open(args.path, 'rb') as f:
        packed_dataset = pickle.load(f)
    logger.info('Total: %d' % len(packed_dataset))

    torch.multiprocessing.set_start_method('spawn')
    # Evaluator
    evaluator = CovMatEvaluator(
        num_workers = args.num_workers,
        ratio = args.ratio,
        print_fn=logger.info,
        use_force_field=args.use_force_field
    )
    results = evaluator(
        packed_data_list = list(packed_dataset),
        start_idx = args.start_idx,
    )
    df_cov, df_mat = print_covmat_results(results, print_fn=logger.info)

    # Save results
    csv_fn_cov = args.path[:-4] + f'_UseFF={args.use_force_field}'+ '_cov.csv'
    csv_fn_mat = args.path[:-4] + f'_UseFF={args.use_force_field}'+ '_mat.csv'
    # csv_fn_cov05mat = args.path[:-4] + '_cov05mat.csv'
    print(csv_fn_cov)
    df_cov.to_csv(csv_fn_cov)
    df_mat.to_csv(csv_fn_mat)
    results2latex(f'{args.path[:-4]}_latex.txt', df_cov, df_mat)
    
    results_fn = args.path[:-4] + f'_UseFF={args.use_force_field}'+ '_covmat.pkl'
    with open(results_fn, 'wb') as f:
        pickle.dump(results, f)
