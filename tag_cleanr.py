"""统计 Tags 数量并且清除 top_n 且在预设 set 的 tags"""

from typing import List, Union, Tuple
from collections import Counter
from pathlib import Path

import json


CHARACTER_FEATURES = {
    "blonde hair",
    "brown hair",
    "black hair",
    "blue hair",
    "pink hair",
    "purple hair",
    "green hair",
    "red hair",
    "silver hair",
    "white hair",
    "grey hair",
    "long hair",
    "short hair",
    "medium hair",
    "twintails",
    "ponytail",
    "braid",
    "ahoge",
    "bangs",
    "blue eyes",
    "red eyes",
    "brown eyes",
    "green eyes",
    "purple eyes",
    "yellow eyes",
    "black eyes",
    "grey eyes",
    # "heterochromia",
    "elf",
    "pointy ears",
    "horns",
    "tail",
    "wings",
    "tan",
    "dark skin",
}

CHARACTER_OUTFIT = {
    # Tops
    "shirt",
    "collared shirt",
    "t-shirt",
    "blouse",
    "sailor collar",
    "sweater",
    "turtleneck",
    "hoodie",
    "tank top",
    "crop top",
    "tube top",
    # Bottoms
    "skirt",
    "pleated skirt",
    "miniskirt",
    "long skirt",
    "pants",
    "jeans",
    "shorts",
    "bike shorts",
    "leggings",
    # Full Body
    "dress",
    "sundress",
    "long dress",
    "jumpsuit",
    "bodysuit",
    "leotard",
    # Outerwear
    "jacket",
    "coat",
    "cardigan",
    "cape",
    "blazer",
    "vest",
    # Hosiery
    "socks",
    "knee-high socks",
    "thighhighs",
    "pantyhose",
    "stockings",
    "leg warmers",
    "fishnets",
    # Footwear
    "boots",
    "knee-high boots",
    "thigh-high boots",
    "shoes",
    "sneakers",
    "heels",
    "sandals",
    "loafers",
    # Headwear
    "hat",
    "beanie",
    "beret",
    "witch hat",
    "hair ribbon",
    "headband",
    "hairpin",
    "hairclip",
    # Accessories
    "gloves",
    "belt",
    "scarf",
    "tie",
    "bow",
    "necktie",
    "choker",
    "glasses",
    "earrings",
    "garter straps",
    "apron",
    # Specific Outfits
    "school uniform",
    "sailor suit",
    "maid outfit",
    "kimono",
    "yukata",
    "swimsuit",
    "bikini",
    "pajamas",
    "armor",
}

QUALITY_TAGS = {
    "masterpiece",
    "best quality",
    "high quality",
    "ultra-detailed",
    "absurdres",
    "highres",
    "4k",
    "8k",
}

STYLE_ARTIST_TAGS = {
    "anime style",
    "manga",
    "realistic",
    "sketch",
    "monochrome",
    "grayscale",
}

COMMON_TAGS = {
    'black hat',
    'hair between eyes',
    'red bow',
    'long sleeves',
    'black jacket',
    'bowtie',
    'red bowtie',
    'white shirt',
    'black skirt',
    'pleated skirt',
    'open clothes',
    'striped bowtie',
    'collared shirt',
    'striped bow',
    'breasts',
    'open jacket',
    'single braid',
    'black socks',
    'earrings',
    'very long hair',
    'jewelry',
    'twin braids',
    'black shoes,',
    'large breast',
}

try:
    with open('removed_tags.json', 'r') as f:
        removed_tags_set = json.load(f)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    removed_tags_set = set()

REMOVE_TAGS = {
    'pink hair', 'long hair', 'choker', 'white choker', 'collarbone', 'breasts', 'yellow eyes', 'white bikini', 'frills', 
    'cleavage', 'large breasts', 'swimsuit', 'bikini', 'navel', 'stomach', 'wings', 'frilled bikini', 'bare shoulders', 
    'underwear', 'bra', 'white bra', 'animal ears', 'fox ears', 'fox girl', 'brown hair', 'animal ear fluff', 'short hair', 
    'official alternate costume', 'visor cap', 'striped bikini', 'medium breasts', 'striped clothes', 'scarf', 'tail', 
    'fox tail', 'shorts', 'highleg', 'highleg bikini', 'bikini under clothes', 'denim shorts', 'denim', 'short shorts', 
    'flower', 'blue shorts', 'yellow nails', 'shirt', 'white shirt', 'long sleeves', 'two side up', 'hair ornament', 
    'dress', 'ribbon', 'school uniform', 'grey hair', 'neck ribbon', 'very long hair', 'hair between eyes', 'sleeveless dress', 
    'collared shirt', 'sleeveless', 'black choker', 'pantyhose', 'buttons', 'orange eyes', 'hair flower', 'black shoes', 
    'rose', 'skirt', 'red rose', 'pinafore dress', 'red flower', 'multicolored hair', 'black hair', 'two-tone hair', 
    'white hair', 'red eyes', 'horns', 'cardigan', 'black horns', 'demon horns', 'black dress', 'black nails', 
    'open clothes', 'nail polish', 'off shoulder', 'black footwear', 'alternate costume', 'jewelry', 'small breasts', 
    'necklace', 'jacket', 'open cardigan', 'earrings', 'cat ears', 'serafuku', 'sailor collar', 'cat tail', 'cat girl', 
    'black sailor collar', 'pleated skirt', 'black eyes', 'extra ears', 'black skirt', 'double bun', 'hair bun', 
    'pink eyes', 'bracelet', 'necktie', 'fake horns', 'red necktie', 'shoes', 'socks', 'thigh strap', 'white socks', 
    'fingernails', 'twintails', 'hat', 'beret', 'yellow dress', 'puffy sleeves', 'blue eyes', 'frilled dress', 
    'short sleeves', 'puffy short sleeves', 'white bow', 'frilled sleeves', 'hood', 'medium hair', 'hooded jacket', 
    'open jacket', 'thighhighs', 'black thighhighs', 'hairclip', 'hood down', 'hairband', 'black hairband', 'gloves', 
    'black gloves', 'virtual youtuber', 'bandaid', 'bandaid on face', 'neckerchief', 'bandaid on cheek', 'blue hair', 
    'midriff peek', 'loose socks', 'bandaid on leg', 'bandages', 'streaked hair', 'bow', 'bandaid on arm', 'midriff', 
    'ahoge', 'white thighhighs', 'hair bow', 'bandaged arm', 'colored inner hair', 'crop top', 'chinese clothes', 
    'red dress', 'red hair', 'china dress', 'antenna hair', 'mole under eye', 'black socks', 'sneakers', 'kneehighs', 
    'mole', 'pelvic curtain', 'wristwatch', 'blue necktie', 'hair ribbon', 'fingerless gloves', 'blazer', 'plaid skirt', 
    'plaid clothes', 'blue ribbon', 'bike shorts', 'blue jacket', 'miniskirt', 'black jacket', 'knee pads', 'black shorts', 
    'sidelocks', 'loafers', 'grey eyes', 'blue skirt', 'grey jacket', 'round eyewear', 'blunt bangs', 'black bow', 
    'black bowtie', 'blush', 'grey skirt', 'mary janes', 'bob cut', 'shirt tucked in', 'red gemstone', 'brooch', 'black-framed eyewear',
}.union(removed_tags_set)

WHITELIST = {
    '1girl',
    'solo',
    'simple background',
    'white background',
    'looking at viewer',
    'full body',
    'upper body',
    'sitting',
    'standing',
    'lying',
    'smile',
    'open mouth',
    'closed mouth',
    'closed eyes',
    'grin',
    'teeth',
    'upper teeth only',
    'holding',
    'crossed arms',
    'arm up',
    'blurry',
    'blurry background',
    'outdoors',
    'blood',
    'blood on face',
    'blood on clothes'
}


BLACKLIST = (
    CHARACTER_FEATURES.union(QUALITY_TAGS).union(STYLE_ARTIST_TAGS).union(COMMON_TAGS).union(REMOVE_TAGS)
    # .union(CHARACTER_OUTFIT)
)


class TagCleaner(object):
    def __init__(self):
        self.counter = Counter()
        self.tags_info = {}
        self.file_count = 0


    def add_tags(self, filename: str, tags: Union[str, List[str]]) -> str:
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]

        self.counter.update(tags)

        if filename in self.tags_info:
            print(f"Warn: Duplicated filename {filename}")
            filename = filename + "-1"

        self.tags_info[filename] = tags
        self.file_count += 1

    def get_cleaned_tags(
        self, top_n: int, clothing_trheshold: int = 0.8
    ) -> List[Tuple[Path, List[str]]]:
        """:param: top_n most_common list 中数量 > top_n 的元素"""
        most_common = self.counter.most_common()
        most_common_top_n = [item for item in most_common if item[1] >= top_n]
        most_common_tags = [a[0] for a in most_common_top_n]
        most_common_tags_count = {i[0]: i[1] for i in most_common_top_n}

        def is_outfit_tags(tag):
            return (
                tag in CHARACTER_OUTFIT
                and most_common_tags_count.get(tag, -1)
                >= self.size * clothing_trheshold
            )

        def is_blacked_tags(tag):
            return tag in most_common_tags and tag in BLACKLIST

        def is_delete_tags(tag):
            """是否是自动删除的 tags"""
            return (is_outfit_tags(tag) or is_blacked_tags(tag))

        # for tag, count in most_common:
        #     if not is_delete_tags(tag):
        #         print(f'{tag} {count}')

        # 其它高频 tags
        common_percent = 0.3
        other_common_tags = [(tag, count) for (tag, count) in most_common if count >= self.file_count * common_percent and not is_delete_tags(tag) and tag not in WHITELIST]
        if other_common_tags:
            print(f'Other common tags(>={common_percent * 100}%):')
            print(', '.join([f'{tag} {count}/{self.file_count}' for tag, count in other_common_tags]))
            print('-' * 20)
            print(', '.join([item[0] for item in other_common_tags]))
        else:
            print(f'No not removed tags >={common_percent * 100}% found')

        user_selected_delete_tags = [tag.strip() for tag in input('Select tags to delete: ').split(',')]
        
        print(f"Removing TOP {top_n} MOST COMMON && BLACKLIST tags:")

        removed_tags_set = set(REMOVE_TAGS)
        for tag, count in most_common_top_n:
            if is_delete_tags(tag) or tag in user_selected_delete_tags:
                print(f"Remove {tag}, count: {count}")
                removed_tags_set.add(tag)

        with open('removed_tags.json', 'w') as f:
            json.dump(list(removed_tags_set), f)

        folder = list(self.tags_info.keys())[0].parent.name.replace('_waifuc', '')

        active_token = input(f'Active token({folder}): ') or folder
        active_token = active_token.replace('_', ' ') 
        
        for filename, tags in self.tags_info.items():
            self.tags_info[filename] = [active_token] + [tag for tag in tags if tag not in removed_tags_set]

        return [(Path(filename), tags) for filename, tags in self.tags_info.items()]

    @property
    def size(self):
        return len(self.tags_info.keys())
