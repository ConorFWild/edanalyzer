import argparse
import os
import shutil
import pathlib
import subprocess
import argparse
import re

from rich import print as rprint
from joblib import Parallel, delayed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', )
    parser.add_argument('--smile_source_file_pattern', default='compound/dep/[a-zA-Z0-9]+.cif', required=False)
    parser.add_argument('--smile_pattern', default='#   SMILES string: ([^\n]+)', required=False)
    parser.add_argument('--compound_dir_pattern', default='compound', required=False)
    parser.add_argument('--dep_pattern', default='_dep', required=False)
    parser.add_argument('--output_pattern', default='compound', required=False)
    parser.add_argument('--dry', default=False, required=False)

    args = parser.parse_args()

    return (
        pathlib.Path(args.path),
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
    match = re.search(smile_pattern, text)
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


def _run_script(script):
    p = subprocess.Popen(script, shell=True, )
    p.communicate()


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
    rprint(f'Grade runscript is: {grade_runscript}')

    # Run
    if not dry:
        _run_script(grade_runscript)


def process_dataset(dataset_dir, smile_source_file_pattern, smile_pattern, compound_dir_pattern, dep_pattern,
                    output_pattern, dry):
    rprint(f'Processing dataset dir: {dataset_dir}')

    if (dataset_dir / dep_pattern).exists():
        return

    # Get smiles
    # rprint([x for x in dataset_dir.rglob('*')])
    smiles_files = [x for x in dataset_dir.rglob('*') if
                    re.match(smile_source_file_pattern, str(x.relative_to(dataset_dir)))]
    if len(smiles_files) == 0:
        rprint(f'Skipping dir: no smiles files!')
        return
    smiles = _parse_file_for_smiles(smiles_files[0], smile_pattern)
    rprint(f'Found smiles: {smiles}')

    # Clean folder
    _clean_folder(dataset_dir / compound_dir_pattern, dataset_dir / dep_pattern, dry)
    rprint(f'Moved {dataset_dir / compound_dir_pattern} to {dataset_dir / dep_pattern}')

    # Generate new cifs
    _generate_ligand_files(smiles, dataset_dir / output_pattern, dry)


def main():
    # Parse Args
    # Walk directory tree
    # # Get smiles from parsing designated source file
    # # Clean directory by moving existing files into a dep folder
    # # Generate new cif by running grade

    # Parse Args
    path, smile_source_file_pattern, smile_pattern, compound_dir_pattern, dep_pattern, output_pattern, dry = _parse_args()

    # Walk Tree
    Parallel(n_jobs=36)(
        delayed(process_dataset)(
            dataset_dir,
            smile_source_file_pattern,
            smile_pattern,
            compound_dir_pattern,
            dep_pattern,
            output_pattern,
            dry,
        )
        for dataset_dir
        in path.glob('*')
    )


if __name__ == "__main__":
    main()
