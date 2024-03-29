import tables


class Annotation(tables.IsDescription):
    epoch = tables.Int32Col()
    idx = tables.Int32Col()
    y = tables.Float32Col()
    y_hat = tables.Float32Col()
    set = tables.Int32Col()


class BuildAnnotation(tables.IsDescription):
    idx = tables.Int32Col()
    event_map_table_idx = tables.Int32Col()
    annotation = tables.BoolCol()
    partition = tables.StringCol(32)


class BuildCorrelation(tables.IsDescription):  #*
    # Calculate the difference between the observed and human
    # modelled density at all atomic coordinates and
    #
    idx = tables.Int32Col()


class EventMapSample(tables.IsDescription):
    idx = tables.Int32Col()
    event_idx = tables.Int32Col()
    res_id = tables.StringCol(32)

    sample = tables.Float32Col(shape=(90, 90, 90))


class MTZSample(tables.IsDescription):
    idx = tables.Int32Col()
    event_idx = tables.Int32Col()
    res_id = tables.StringCol(32)

    sample = tables.Float32Col(shape=(90, 90, 90))


class PoseSample(tables.IsDescription):
    idx = tables.Int32Col()

    # = tables.StringCol(length=64)
    database_event_idx = tables.Int32Col()

    positions = tables.Float32Col(shape=(60, 3,))
    elements = tables.Int32Col(shape=(60,))
    rmsd = tables.Float32Col()

    event_map_sample_idx = tables.Int32Col()
    mtz_sample_idx = tables.Int32Col()


class Delta(tables.IsDescription):
    idx = tables.Int32Col()

    pose_idx = tables.Int32Col()

    delta = tables.Float32Col(shape=(60,))
    delta_vec = tables.Float32Col(shape=(60, 3,))
