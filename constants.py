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
PANNDA_ANNOTATIONS_FILE = "pandda_annotations.json"
PANDDA_DATA_ROOT_DIR = "/dls/labxchem/data"
DIAMOND_PROCESSING_DIR = "processing"
DIAMOND_ANALYSIS_DIR = "analysis"
DIAMOND_MODEL_BUILDING_DIR_NEW = "model_building"
DIAMOND_MODEL_BUILDING_DIR_OLD = "initial_model"

PANDDA_ANALYSIS_DIR = "analyses"
PANDDA_INSPECT_TABLE_FILE = "pandda_inspect_events.csv"
PANDDA_PROCESSED_DATASETS_DIR = "processed_datasets"
PANDDA_INSPECT_MODEL_DIR = "modelled_structures"
PANDDA_EVENT_MAP_TEMPLATE = "{dtag}-event_{event_idx}_1-BDC_{bdc}_map.native.ccp4"
PANDDA_MODEL_FILE = "{dtag}-pandda-model.pdb"
PANDDA_INITIAL_MODEL_TEMPLATE = "{dtag}-pandda-input.pdb"

PANDDA_INSPECT_DTAG = "dtag"
PANDDA_INSPECT_EVENT_IDX = "event_idx"
PANDDA_INSPECT_BDC = "1-BDC"
PANDDA_INSPECT_X = "x"
PANDDA_INSPECT_Y = "y"
PANDDA_INSPECT_Z = "z"
PANDDA_INSPECT_HIT_CONDFIDENCE = "Ligand Confidence"
PANDDA_INSPECT_TABLE_HIGH_CONFIDENCE = "High"
