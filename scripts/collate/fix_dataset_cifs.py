import argparse
import os
import shutil
import pathlib
import subprocess
import argparse
import re

from rich import print as rprint


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', )
    parser.add_argument('--smile_source_file_pattern', default='/compound/dep/[a-zA-Z0-9]+.cif', required=False)
    parser.add_argument('--smile_pattern', default='#   SMILES string: ([^\n])', required=False)
    parser.add_argument('--compound_dir_pattern', default='compound', required=False)
    parser.add_argument('--dep_pattern', default='_dep', required=False)
    parser.add_argument('--output_pattern', default='compound', required=False)
    parser.add_argument('--dry', default=True, required=False)

    args = parser.parse_args()

    return (
        args.path,
        args.smile_source_file_pattern,
        args.smile_pattern,
        args.compound_dir_pattern,
        args.dep_pattern,
        args.output_pattern,
        args.dry
    )


def _parse_file_for_smiles(smile_source_file_pattern, smile_pattern):
    # Read file
    with open(smile_source_file_pattern, 'r') as f:
        text = f.read()

    # Parse with RE
    match = re.search(smile_pattern, smile_source_file_pattern)
    return match.group(1)


def _clean_folder(compound_dir, dep_dir, dry=True):
    # move the compound dir to the dep dir
    if not dry:
        shutil.move(compound_dir, dep_dir)


def _write_smiles(smiles, smiles_output_path):
    with open(smiles_output_path, 'w') as f:
        f.write(smiles)


def _get_grade_runscript(output_pattern, smiles_output_path, ):
    runscript = f"cd {output_pattern}; module load buster; grade -in {smiles_output_path}"
    return runscript


def _generate_ligand_files(smiles, output_pattern, dry):
    # Create output dir
    if not dry:
        os.mkdir(output_pattern)
    rprint(f'Made output directory: {output_pattern}')

    # Write smiles
    smiles_output_path = output_pattern / 'grade-XXX.smiles'
    if not dry:
        _write_smiles(smiles, smiles_output_path)
    rprint(f'Wrote smiles {smiles} to file {smiles_output_path}')

    # Generate script to run
    grade_runscript = _get_grade_runscript(output_pattern, smiles_output_path, )

    # Run
    run_script(grade_runscript)

    ...


def main():
    # Parse Args
    # Walk directory tree
    # # Get smiles from parsing designated source file
    # # Clean directory by moving existing files into a dep folder
    # # Generate new cif by running grade

    # Parse Args
    path, smile_source_file_pattern, smile_pattern, compound_dir_pattern, dep_pattern, output_pattern, dry = _parse_args()

    # Walk Tree
    for dataset_dir in path.glob('*'):
        rprint(f'Processing dataset dir: {dataset_dir}')

        # Get smiles
        smiles = _parse_file_for_smiles(dataset_dir / smile_source_file_pattern, smile_pattern)
        rprint(f'Found smiles: {smiles}')

        # Clean folder
        _clean_folder(dataset_dir / compound_dir_pattern, dataset_dir / dep_pattern, dry)
        rprint(f'Moved {dataset_dir / compound_dir_pattern} to {dataset_dir / dep_pattern}')

        # Generate new cifs
        _generate_ligand_files(smiles, dataset_dir / output_pattern, dry)


if __name__ == "__main__":
    main()
