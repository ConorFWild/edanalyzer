import tables
import yaml

from rich import print as rprint

class Annotation(tables.IsDescription):
    epoch = tables.Int32Col()
    idx = tables.Int32Col()
    y = tables.Float32Col()
    y_hat = tables.Float32Col()


if __name__ == "__main__":
    rprint('Reading yaml...')
    with open('./output/build_scoring/annotations_train.yaml', 'r') as f:
        annotations = yaml.safe_load(f)

    rprint(f'Creating table...')
    # Open a file in "w"rite mode
    fileh = tables.open_file("objecttree.h5", mode="w")

    # Get the HDF5 root group
    root = fileh.root

    # Create 2 new tables in group1
    rprint(f"Creating table")
    table1 = fileh.create_table(root, "test_annotations", Annotation)

    # insert the annotations
    annotation = table1.row

    for _epoch, _records in annotations['test'].items():
        rprint(f"Adding {len(_records)} annotations for epoch: {_epoch}")
        for _annotation_set in _records:
            for _annotation in _annotation_set:
                annotation['epoch'] = int(_epoch)
                annotation['idx'] = int(_annotation['idx'])
                annotation['y'] = float(_annotation['y'])
                annotation['y_hat'] = float(_annotation['y_hat'])
                annotation.append()
    table1.flush()
    rprint(f"Table flushed")
    fileh.close()
    rprint("Table closed")

