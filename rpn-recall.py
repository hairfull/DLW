import detectron2
from detectron2.config import get_cfg
from detectron2.structures import Boxes, pairwise_iou
from detectron2.engine import launch
import numpy as np

from defrcn.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.dataloader import build_detection_test_loader
from defrcn.engine import default_argument_parser, default_setup, DefaultPredictor
from tqdm import tqdm
import tools.create_config
from defrcn.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from main import Trainer


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    # cfg.MODEL.WEIGHTS = '/home/wxq/od/DeFRCN/checkpoints/new_dynamic/pth/3/5shot_seed1/model_final.pth'
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = Trainer.build_model(cfg)
    model.eval()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    Trainer.calc_recall(cfg, model)
    return


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


