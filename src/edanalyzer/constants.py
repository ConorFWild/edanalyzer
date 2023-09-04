DATA_DIR = "data"
OPTIONS_FILE = "options.json"
DATASET_FILE = "dataset.json"
RSYNC_COMMAND = "rsync -rlpt -v -z --delete rsync.ebi.ac.uk::pub/databases/pdb/data/{datatype_dir}/ \"{data_dir}/{datatype_dir}/\""
STRUCTURE_FACTORS = (
    ('pdbx_FWT', 'pdbx_PHWT'),
    ("FWT", "PHWT"),
    ("2FOFCWT", "PH2FOFCWT"),
    ("2FOFCWT_iso-fill", "PH2FOFCWT_iso-fill"),
    ("2FOFCWT_fill", "PH2FOFCWT_fill",),
    ("2FOFCWT", "PHI2FOFCWT"),
)
PDB_REGEX = "pdb([a-zA-Z0-9]+)\.ent\.gz"
MTZ_REGEX = "r([a-zA-Z0-9]+)sf\.ent\.gz"

PANDDA_DATASET_FILE = "pandda_dataset.json"
TRAIN_SET_FILE = "pandda_dataset_train.json"
TRAIN_SET_ANNOTATION_FILE = "pandda_annotations_train.json"
TEST_SET_FILE = "pandda_dataset_test.json"
FINETUNE_TRAIN_EVENTS_FILE = "finetune_train_events.json"
FINETUNE_TRAIN_SET_ANNOTATION_FILE = "finetune_train_annotations.json"
TEST_SET_ANNOTATION_FILE = "pandda_annotations_test.json"
PANNDA_ANNOTATIONS_FILE = "pandda_annotations.json"
PANDDA_DATA_ROOT_DIR = "/dls/labxchem/data"
DIAMOND_PROCESSING_DIR = "processing"
DIAMOND_ANALYSIS_DIR = "analysis"
DIAMOND_MODEL_BUILDING_DIR_NEW = "model_building"
DIAMOND_MODEL_BUILDING_DIR_OLD = "initial_model"
PANDDA_UPDATED_EVENT_ANNOTATIONS_FILE = "pandda_updated_event_annotations.json"
PANDDA_TEST_ANNOTATION_DIR = "TEST"
PANDDA_TRAIN_ANNOTATION_DIR = "TRAIN"

PANDDA_UPDATED_TEST_EVENT_ANNOTATIONS_FILE = "pandda_updated_test_event_annotations.json"
SQLITE_FILE = "database.db"

# MODEL_FILE = "model_state_dict_zero.pt"
MODEL_FILE = "model_state_dict_zero_z.pt"
MODEL_FILE_EPOCH = "model_state_dict_{epoch}.pt"
MODEL_FILE_REGEX = "model_state_dict_([0-9]+).pt"

MODEL_FILE_EPOCH_XMAP_MEAN = "model_state_dict_xmap_mean_{epoch}.pt"
MODEL_FILE_REGEX_XMAP_MEAN = "model_state_dict_xmap_mean_([0-9]+).pt"

MODEL_FILE_EPOCH_XMAP_LIGAND = "model_state_dict_xmap_ligand_{epoch}.pt"
MODEL_FILE_REGEX_XMAP_LIGAND = "model_state_dict_xmap_ligand_([0-9]+).pt"

PANDDA_ANALYSIS_DIR = "analyses"
PANDDA_INSPECT_TABLE_FILE = "pandda_inspect_events.csv"
PANDDA_PROCESSED_DATASETS_DIR = "processed_datasets"
PANDDA_INSPECT_MODEL_DIR = "modelled_structures"
PANDDA_EVENT_MAP_TEMPLATE = "{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"
PANDDA_MODEL_FILE = "{dtag}-pandda-model.pdb"
PANDDA_INITIAL_MODEL_TEMPLATE = "{dtag}-pandda-input.pdb"
PANDDA_INITIAL_MTZ_TEMPLATE = "{dtag}-pandda-input.mtz"
PANDDA_EVENT_TABLE_PATH = "pandda_analyse_events.csv"
PANDDA_SITE_TABLE_PATH = "pandda_analyse_sites.csv"
PANDDA_ZMAP_TEMPLATE = "{dtag}-z_map.native.ccp4"
PANDDA_GROUND_STATE_MAP_TEMPLATE = "{dtag}-ground-state-average-map.native.ccp4"

PANDDA_INSPECT_DTAG = "dtag"
PANDDA_INSPECT_EVENT_IDX = "event_idx"
PANDDA_INSPECT_BDC = "1-BDC"
PANDDA_INSPECT_X = "x"
PANDDA_INSPECT_Y = "y"
PANDDA_INSPECT_Z = "z"
PANDDA_INSPECT_HIT_CONDFIDENCE = "Ligand Confidence"
PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE = "High"
PANDDA_INSPECT_TABLE_LOW_CONFIDENCE = "Low"
PANDDA_INSPECT_VIEWED = "Viewed"
PANDDA_INSPECT_SITE_IDX = "site_idx"

HIGH_SCORING_NON_HIT_DATASET_DIR = "HIGH_SCORING_NON_HIT_DATASET_DIR"
LOW_SCORING_HIT_DATASET_DIR = "LOW_SCORING_HIT_DATASET_DIR"

MODEL_BUILDING_STRUCTURE_FILE = "dimple.pdb"
MODEL_BUILDING_REFLECTIONS_FILE = "dimple.mtz"
OLD_MODEL_BUILDING_STRUCTURE_FILE_TEMPLATE = "{dtag}.dimple.pdb"
OLD_MODEL_BUILDING_REFLECTIONS_FILE_TEMPLATE = "{dtag}.dimple.mtz"



TABLE_EVENT_PARTITION = "event_partition_association_table"
TABLE_DATASET_PANDDA = "dataset_pandda_association_table"
TABLE_PANDDA = "pandda"
TABLE_EVENT = "event"
TABLE_DATASET = "dataset"
TABLE_LIGAND = "ligand"
TABLE_ANNOTATION = "annotation"
TABLE_PARTITION = "partition"
TABLE_SYSTEM = "system"
TABLE_EXPERIMENT = "experiment"

INITIAL_TRAIN_PARTITION = "train"
INITIAL_TEST_PARTITION = "test"
FINETUNE_TRAIN_PARTITION = "finetune_train"
FINETUNE_TEST_PARTITION = "finetune_test"

LIGAND_IGNORE_REGEXES = [
    "merged",
    "LIG-[a-zA-Z]+-",
    "dimple",
    "refine",
    "init",
    "pipedream",
    "phenix",
    "None",
    "blank",
    "control",
    "DMSO",
]