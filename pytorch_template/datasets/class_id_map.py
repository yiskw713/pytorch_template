from typing import Dict


def get_cls2id_map() -> Dict[str, int]:
    cls2id_map = {"daisy": 0, "dandelion": 1, "rose": 2, "sunflower": 3, "tulip": 4}

    return cls2id_map


def get_id2cls_map() -> Dict[int, str]:
    cls2id_map = get_cls2id_map()
    return {val: key for key, val in cls2id_map.items()}
