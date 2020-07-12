import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--dataset', type=str, default='Reuters', choices=['Reuters', 'CongressionalHearing', 'AAPD', 'IMDB', 'Yelp2014'])
    parser.add_argument('--max-vocab-size', type=int, default=500000)
    parser.add_argument('--max-vocab-sizes', nargs='+', type=int,  
                        help='Vocab sizes for each fold when doing cross-validation')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'lr'))
    parser.add_argument('--resume-snapshot', type=str)
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
