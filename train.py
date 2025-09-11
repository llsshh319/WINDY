import warnings
# warnings.filterwarnings('ignore')

from utils import get_dist_info
from parser import create_parser
from exp import BaseExperiment


if __name__ == '__main__':
    args = create_parser().parse_args()

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if args.test:
        if rank == 0:
            print('>'*35 + ' testing  ' + '<'*35)
        exp.test()
