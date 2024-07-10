import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("rdd_test_1",          "RDD", "test",      "base_novel_1", 1),
        ("rdd_test_2",          "RDD", "test",      "base_novel_2", 2),
        ("rdd_test_3",          "RDD", "test",      "base_novel_3", 3),
        ("rdd_trainval_base_1", "RDD", "trainval",  "base1",        1),
        ("rdd_trainval_base_2", "RDD", "trainval",  "base2",        2),
        ("rdd_trainval_base_3", "RDD", "trainval",  "base3",        3),
        ("laf_test_1",          "LAF", "test",      "base_novel_1", 1),
        ("laf_test_2",          "LAF", "test",      "base_novel_2", 2),
        ("laf_test_3",          "LAF", "test",      "base_novel_3", 3),
        ("laf_trainval_base_1", "LAF", "trainval",  "base1",        1),
        ("laf_trainval_base_2", "LAF", "trainval",  "base2",        2),
        ("laf_trainval_base_3", "LAF", "trainval",  "base3",        3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for dataset in ['rdd', 'laf']:
                    for seed in range(20):
                        seed = "_seed{}".format(seed)
                        name = "{}_trainval_{}{}_{}shot{}".format(
                            dataset, prefix, sid, shot, seed
                        )
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dataset.upper(), img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        # year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# register_all_coco()
register_all_voc()