from models.bert_hier.__main__ import run_main
from models.bert_hier.args import get_args
from utils.postprocessing import process_json_results

if __name__ == '__main__':
    args = get_args()

    if args.num_folds < 2:
        raise ValueError("Number of folds must be greater than 1!", args.num_folds)

    orig_metrics_json = args.metrics_json
    for fold in range(0, args.num_folds):
        print('On fold', str(fold))
        num_train_restarts = 0
        args.fold_num = fold
        orig_seed = args.seed
        if orig_metrics_json:
            args.metrics_json = orig_metrics_json + '_fold' + str(fold)
        training_converged = run_main(args)
        while not training_converged and num_train_restarts < args.num_train_restarts:
            num_train_restarts += 1
            args.seed += 10
            print('Rerunning fold', fold, 'with new seed', args.seed)
            training_converged = run_main(args)
        args.seed = orig_seed
    # summarize fold results and save to file
    process_json_results(orig_metrics_json, orig_metrics_json + '_fine_summary.tsv', 'test', label_suffix='_fine')
    process_json_results(orig_metrics_json, orig_metrics_json + '_coarse_summary.tsv', 'test', label_suffix='_coarse')

