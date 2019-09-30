def get_id2label_map():
    cls2id_map = {
        'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4
    }

    return cls2id_map


def get_label2id_map():
    cls2id_map = get_id2label_map()
    return {val: key for key, val in cls2id_map.items()}
