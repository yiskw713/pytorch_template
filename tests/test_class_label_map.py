from src.libs.class_id_map import get_cls2id_map, get_id2cls_map


def test_get_cls2id_map() -> None:
    cls2id_map = get_cls2id_map()

    assert len(cls2id_map) == 5
    assert cls2id_map["daisy"] == 0


def test_get_id2cls_map() -> None:
    id2cls_map = get_id2cls_map()

    assert len(id2cls_map) == 5
    assert id2cls_map[0] == "daisy"
