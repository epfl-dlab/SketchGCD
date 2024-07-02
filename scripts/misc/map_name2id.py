import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.const import (
    KG_catalogue_DIR,
    PATH_TO_WIKI_ENTITY2ID_MAPPING,
    PATH_TO_WIKI_RELATION2ID_MAPPING,
)
from src.constrained_generation.entity_trie_preparation import (
    get_ids_for_names,
    read_constrained_world,
)


if __name__ == "__main__":
    constrained_world_name = "wikinre"
    entities, relations = read_constrained_world(constrained_world_name)

    original_num_entities = len(entities)
    original_num_relations = len(relations)
    rel_ids = get_ids_for_names(relations, PATH_TO_WIKI_RELATION2ID_MAPPING)
    ent_ids = get_ids_for_names(entities, PATH_TO_WIKI_ENTITY2ID_MAPPING)
    print(
        f"Original number of entities: {original_num_entities} and relations: {original_num_relations}"
    )
    print(f"Mapped number of entities: {len(ent_ids)} and relations: {len(rel_ids)}")
    """
    Original number of entities: 278843 and relations: 158
    Mapped number of entities: 67443 and relations: 158
    """
