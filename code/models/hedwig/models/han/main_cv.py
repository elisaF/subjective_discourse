from models.han.__main__ import run_main
from models.han.args import get_args
from utils.postprocessing import process_json_results

if __name__ == '__main__':
    args = get_args()

    if args.num_folds < 2:
        raise ValueError("Number of folds must be greater than 1!", args.num_folds)

    orig_metrics_json = args.metrics_json
    for fold in range(0, args.num_folds):
        print('On fold', str(fold))
        args.fold_num = fold
        if orig_metrics_json:
            args.metrics_json = orig_metrics_json + '_fold' + str(fold)
        run_main(args)

    # summarize fold results and save to file
    process_json_results(orig_metrics_json, orig_metrics_json+'_summary.tsv', 'test')



