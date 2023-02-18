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