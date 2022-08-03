from evaluate import mc_matrices
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='metric name')
    args = parser.parse_args()
    print('Computing corss evaluations....')
    mc_matrices(args.type,'1658867433_10', '1658878251_20', '1658878306_1', '1658878329_10', '1658878488_20', '1658878564_40', '1658879092_1', '1658879962_10', '1658880548_20', '1658880677_40', '1658880769_1', '1658880919_10', '1658880930_20', '1658881391_40', '1658915426_1', '1658915465_10', '1658915507_1', '1658915507_20', '1658915507_40', '1658915560_10', '1658917814_20', '1658917828_1', '1658917828_40', '1658917929_10', '1658917970_20', '1658917984_40', '1658918615_1', '1658920025_10', '1658920182_20', '1658920199_40', '1658920299_1', '1658920332_10', '1658920350_20', '1658921527_40', '1658922358_1', '1658922582_10', '1658922608_20', '1658922696_40', '1658922742_1', '1658922747_10', '1658924556_20', '1658924908_40')
