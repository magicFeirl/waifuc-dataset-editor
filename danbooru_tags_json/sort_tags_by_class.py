#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Iterable, List

# Class definitions (embedded from danbooru_tags_class_definitions.json)
CLASS_DEFS = [
    {"id": -1, "en": "unknown", "zh": "未知"},
    {"id": 1, "en": "other", "zh": "其他"},
    {"id": 2, "en": "scenery", "zh": "场景"},
    {"id": 3, "en": "appearance", "zh": "外观"},
    {"id": 4, "en": "pose", "zh": "姿态"},
    {"id": 5, "en": "clothing", "zh": "服饰"},
]

# Custom class order by key (use en names); empty means default CLASS_DEFS order
CLASS_ORDER = [
    "pose",
    "appearance",
    "clothing",
    "scenery",
    "other",
    "unknown"
]

BASE_DIR = Path(__file__).resolve().parent
TAGS_PATH = BASE_DIR / "danbooru_tags_exclude_other.json"


def load_tag_classes(path: Path) -> Dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_input_tags(text: str) -> List[str]:
    raw = [t.strip() for t in text.split(',')]
    return [t for t in raw if t]


def build_class_order() -> List[dict]:
    if not CLASS_ORDER:
        return CLASS_DEFS

    en_to_def = {c["en"]: c for c in CLASS_DEFS}
    ordered: List[dict] = [en_to_def[name] for name in CLASS_ORDER if name in en_to_def]
    remaining = [c for c in CLASS_DEFS if c["en"] not in CLASS_ORDER]
    return ordered + remaining


def sort_tags(tags: Iterable[str], tag_to_class: Dict[str, int], class_order: Dict[int, int]) -> List[str]:
    def key(tag: str):
        class_id = tag_to_class.get(tag)
        class_idx = class_order.get(class_id, 10**9)
        return (class_idx, tag)

    return sorted(tags, key=key)


def group_by_class(tags: Iterable[str], tag_to_class: Dict[str, int]) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    for tag in tags:
        # 无法找到的 tags 排序为 -1
        class_id = tag_to_class.get(tag, -1)
        grouped.setdefault(class_id, []).append(tag.replace('_', ' '))
    return grouped


def main() -> int:
    tag_to_class = load_tag_classes(TAGS_PATH)
    ordered_classes = build_class_order()
    class_order_index = {item["id"]: idx for idx, item in enumerate(ordered_classes)}

    while True:
        lines: List[str] = []
        try:
            line = input("Input tags (empty to exit): ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)

        while True:
            try:
                line = input()
            except EOFError:
                line = ""
            if not line.strip():
                break
            lines.append(line)

        input_text = ", ".join(lines).strip()
        if not input_text:
            continue

        input_tags = [t.replace(' ', '_') for t in parse_input_tags(input_text)]

        sorted_tags = sort_tags(input_tags, tag_to_class, class_order_index)
        grouped = group_by_class(sorted_tags, tag_to_class)

        for cls in ordered_classes:
            cid = cls["id"]
            if cid not in grouped:
                continue
            name = cls.get("zh") or cls.get("en") or str(cid)
            grouped[cid].sort()
            line = ", ".join(grouped[cid])
            print(f"{name}: {line}")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
