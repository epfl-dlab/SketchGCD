# const.py
import os

# Define the base directory of your project (optional but often useful)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to various resources and data directories
ASSET_DIR = os.path.join(BASE_DIR, "..", "assets")
PROMPTER_DIR = os.path.join(ASSET_DIR, "prompters")
CONSTRAINTS_DIR = os.path.join(BASE_DIR, "..", "constraints")


KG_DIR = os.path.join(CONSTRAINTS_DIR, "KG")
CACHE_DIR = os.path.join(KG_DIR, ".cache")
KG_catalogue_DIR = os.path.join(KG_DIR, "KG_catalogue")
KG_metadata_DIR = os.path.join(KG_DIR, "KG_metadata")

GRAMMR_DIR = os.path.join(CONSTRAINTS_DIR, "ebnf_grammar")

PATH_TO_WIKI_RELATION2ID_MAPPING = os.path.join(
    KG_metadata_DIR, "wiki_id2name_mappings", "relation_mapping.jsonl"
)
PATH_TO_WIKI_ENTITY2ID_MAPPING = os.path.join(
    KG_metadata_DIR, "wiki_id2name_mappings", "entity_mapping.jsonl"
)
