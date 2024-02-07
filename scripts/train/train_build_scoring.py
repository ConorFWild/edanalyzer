import fire

import lightning as lt

from edanalyzer.datasets.build_scoring import BuildScoringDataset
from edanalyzer.models.build_scoring import LitBuildScoring


def main():
    trainer = lt.Trainer(accelerator='gpu')
    dataset_train = BuildScoringDataset()
    dataset_test = BuildScoringDataset()
    model = LitBuildScoring()
    trainer.fit(model, dataset_train, dataset_test)


if __name__ == "__main__":
    fire.Fire(main)