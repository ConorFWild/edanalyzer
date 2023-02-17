DATA_DIR = "data"
OPTIONS_FILE = "options.json"
DATASET_FILE = "dataset.json"
RSYNC_COMMAND = "rsync -rlpt -v -z --delete rsync.ebi.ac.uk::pub/databases/pdb/data/{datatype_dir}/ \"{data_dir}}/{datatype_dir}/\""
