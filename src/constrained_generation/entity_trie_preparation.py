import jsonlines
import json
import os

from pathlib import Path
from src.utils.IE_linearization_utils import LinearizationType, remove_accents_from_str
from src.const import KG_catalogue_DIR


def get_trie_from_strings(
    string_iterable,
    output_folder_path=None,
    trie_name=None,
    tokenizer=None,
):
    from src.constrained_generation import Trie

    assert (output_folder_path is None) == (
        trie_name is None
    ), f"Both output_folder_path and trie_name should be provided or None. Got {output_folder_path} and {trie_name} respectively."

    from tqdm import tqdm

    if tokenizer is None:
        from transformers import T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    # here we avoid adding special tokens but we add the eos token. Because eos token is needed for the trie to work
    # but we don't want to add special tokens. This should work for all tokenizers except for the ones that
    # don't have an eos token. TODO: check if there is tokenizer that doesn't have an eos token
    encode_func = lambda x: tokenizer(x, add_special_tokens=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    trie = Trie(
        [
            encode_func(uniq_name)
            for uniq_name in tqdm(sorted(string_iterable), desc="Building trie")
        ]
    )

    if output_folder_path is not None:
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        trie.dump(
            output_folder_path=output_folder_path,
            file_name=trie_name,
            string_iterable=string_iterable,
        )

    return trie


def read_constrained_world(constrained_world_name: str):

    constrained_world_dir = os.path.join(KG_catalogue_DIR, constrained_world_name)
    entities_file_path = os.path.join(constrained_world_dir, "entities.json")
    relations_file_path = os.path.join(constrained_world_dir, "relations.json")

    with open(entities_file_path) as json_file:
        entities = set(json.load(json_file))

    with open(relations_file_path) as json_file:
        relations = set(json.load(json_file))

    return entities, relations


def write_constrained_world(
    path_to_constrained_worlds_dir, constrained_world_id, entity_ids, relation_ids
):
    path_to_constrained_world_dir = os.path.join(
        path_to_constrained_worlds_dir, constrained_world_id
    )
    os.makedirs(path_to_constrained_world_dir, exist_ok=True)

    with open(
        os.path.join(path_to_constrained_world_dir, "relations.json"), "w"
    ) as json_file:
        if isinstance(relation_ids, set):
            relation_ids = list(relation_ids)
        json.dump(list(relation_ids), json_file)

    with open(
        os.path.join(path_to_constrained_world_dir, "entities.json"), "w"
    ) as json_file:
        if isinstance(entity_ids, set):
            entity_ids = list(entity_ids)
        json.dump(entity_ids, json_file)


def get_names_for_ids(ids, path_to_id2name_mapping, keep_spaces, remove_accents=False):
    with jsonlines.open(path_to_id2name_mapping) as reader:
        id2name_mapping = {obj["id"]: obj["en_label"] for obj in reader}

    names = [
        LinearizationType.normalize_spaces(
            id2name_mapping[_id], keep_spaces=keep_spaces
        )
        for _id in ids
        if _id in id2name_mapping
    ]

    if remove_accents:
        names = [remove_accents_from_str(name) for name in names]

    return names


def get_ids_for_names(names, path_to_name2id_mapping):
    # with jsonlines.open(path_to_name2id_mapping) as reader:
    #     name2id_mapping = {obj["en_label"]: obj["id"] for obj in reader}
    name2id_mapping = {}
    with jsonlines.open(path_to_name2id_mapping) as reader:
        for i, obj in enumerate(reader):
            try:
                name2id_mapping[obj["en_label"]] = obj["id"]
            except jsonlines.InvalidLineError as e:
                print(f"Invalid JSON at line {i + 1}: {e}")
    return [name2id_mapping[name] for name in names if name in name2id_mapping]


def encode(text, tokenizer, keep_eos: bool):
    if keep_eos:
        raise NotImplementedError
        # return tokenizer.encode(text)

    return tokenizer.encode(text, add_special_tokens=False)
