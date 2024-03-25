import gemmi

def _get_model(closest_pose):
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(45.0, 45.0, 45.0, 90.0, 90.0, 90.0)
    st.spacegroup_hm = gemmi.SpaceGroup('P1').xhm()
    model = gemmi.Model('0')
    chain = gemmi.Chain('A')
    res = gemmi.Residue()
    res.name = 'LIG'
    res.seqid = gemmi.SeqId(1, ' ')

    for _pose_row, _element in zip(closest_pose['positions'], closest_pose['elements']):
        pos = gemmi.Position(_pose_row[0], _pose_row[1], _pose_row[2])
        if _element == 0:
            continue

        element = gemmi.Element(_element)
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.charge = 0
        atom.pos = pos
        atom.element = element
        res.add_atom(atom)

    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st

def _write_structure(st, path):
    st.write_pdb(str(path))
