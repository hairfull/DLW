import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rdd', help='', choices=['rdd'])
    parser.add_argument('--config_root', type=str, default='', help='the path to config dir')
    parser.add_argument('--shot', type=int, default=1, help='shot to run experiments over')
    parser.add_argument('--seed', type=int, default=0, help='seed to run experiments over')
    parser.add_argument('--split', type=int, default=1, help='only for voc')
    args = parser.parse_args()
    return args


def load_config_file(yaml_path):
    fpath = os.path.join(yaml_path)
    yaml_info = open(fpath).readlines()
    return yaml_info


def save_config_file(yaml_info, yaml_path):
    wf = open(yaml_path, 'w')
    for line in yaml_info:
        wf.write('{}'.format(line))
    wf.close()


def main():
    args = parse_args()
    # suffix = 'novel' if args.setting == 'fsod' else 'all'

    # if args.dataset in ['voc']:
    #     # 都是拿x作为template的
    #     name_template = 'defrcn_{}_r101_novelx_{}shot_seedx.yaml'
    #     yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.shot))
    #     yaml_info = load_config_file(yaml_path)
    #     for i, lineinfo in enumerate(yaml_info):
    #         if '  TRAIN: ' in lineinfo:
    #             _str_ = '  TRAIN: ("rdd_trainval_{}{}_{}shot_seed{}", )\n'
    #             yaml_info[i] = _str_.format(suffix, args.split, args.shot, args.seed)
    #         if '  TEST: ' in lineinfo:
    #             _str_ = '  TEST: ("rdd_test",)\n'
    #             yaml_info[i] = _str_
    #     yaml_path = yaml_path.replace('novelx', 'novel{}'.format(args.split))
    # elif args.dataset in ['coco14']:
    #     name_template = 'defrcn_{}_r101_novel_{}shot_seedx.yaml'
    #     yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.shot))
    #     yaml_info = load_config_file(yaml_path)
    #     for i, lineinfo in enumerate(yaml_info):
    #         if '  TRAIN: ' in lineinfo:
    #             _str_ = '  TRAIN: ("coco14_trainval_{}_{}shot_seed{}", )\n'
    #             yaml_info[i] = _str_.format(suffix, args.shot, args.seed)
    # else:
    #     raise NotImplementedError

    if args.dataset == 'rdd':
        name_template = 'DGT_rdd.yaml'
        yaml_path = os.path.join(args.config_root, name_template)
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            # 逆天，设成'train'还真会出错
            if '  TRAIN: ' in lineinfo:
                yaml_info[i] = f'  TRAIN: ("rdd_trainval_novel{args.split}_{args.shot}shot_seed{args.seed}", )\n'
        yaml_path = f'{args.config_root}/DGT_rdd_split{args.split}_{args.shot}shot_seed{args.seed}.yaml'
    else:
        raise NotImplementedError

    save_config_file(yaml_info, yaml_path)


if __name__ == '__main__':
    main()
