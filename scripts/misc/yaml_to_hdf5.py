import pytables
import yaml

class Annotation(pytables.IsDescription):
    epoch = pytables.Int32Col()
    idx = pytables.Int32Col()
    y = pytables.Float32Col()
    y_hat = pytables.Float32Col()


if __name__ == "__main__":
    with open('./output/build_scoring/annotations_train.yaml', 'r') as f:
        annotations = yaml.safe_load(f)

    # Open a file in "w"rite mode
    fileh = pytables.open_file("objecttree.h5", mode="w")

    # Get the HDF5 root group
    root = fileh.root

    # Create 2 new tables in group1
    table1 = fileh.create_table(root, "test_annotations", Annotation)

    # insert the annotations
    annotation = table1.row

    for _epoch, _records in annotations['test'].items():
        for _annotation in _records:
            annotation['epoch'] = _epoch
            annotation['idx'] = _annotation['idx']
            annotation['y'] = _annotation['y']
            annotation['y_hat'] = _annotation['y_hat']
            annotation.append()
    table1.flush()
    fileh.close()
