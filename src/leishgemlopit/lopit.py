from __future__ import annotations
from collections import Counter
from collections.abc import Mapping
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd

from orthomcl import OrthoMCL

DEFAULT_TMT_LABELS = [
    "126",
    "127C",
    "127N",
    "128C",
    "128N",
    "129C",
    "129N",
    "130C",
    "130N",
    "131C",
    "131N",
]


class Gene:
    gene_id: str
    data_entries: dict[LOPITExperiment, LOPITDataEntry]
    description: str | None
    organism: str | None

    _CACHE: dict[str, Gene] = WeakValueDictionary()

    def __init__(self, gene_id: str):
        self.gene_id = gene_id
        self.data_entries = {}
        try:
            orthomcl_entry = OrthoMCL.get_entry(gene_id)
            self.description = orthomcl_entry.description
            self.organism = orthomcl_entry.organism

        except KeyError:
            self.description = None

        if gene_id in Gene._CACHE:
            raise ValueError("Gene is already cached, why am I being created?")
        Gene._CACHE[gene_id] = self

    @staticmethod
    def from_gene_id(gene_id: str):
        try:
            return Gene._CACHE[gene_id]
        except KeyError:
            return Gene(gene_id)

    def __hash__(self):
        return hash(self.gene_id)

    def __eq__(self, other: Gene):
        if not isinstance(other, Gene):
            return NotImplemented
        return hash(self) == hash(other)

    def add_entry(self, entry: LOPITDataEntry):
        if entry.experiment in self.data_entries:
            raise ValueError("Data for this experiment already present.")
        self.data_entries[entry.experiment] = entry


class LOPITDataEntry:
    def __init__(
        self,
        experiment: LOPITExperiment,
        gene: Gene,
        data: np.ndarray,
        labels: list[str] = DEFAULT_TMT_LABELS,
    ):
        self.experiment = experiment
        self.gene = gene
        self.data = data
        self.labels = labels

    def to_series(self):
        return pd.Series(
            self.data,
            index=self.labels,
            name=(self.gene.gene_id, self.experiment.name)
        ).rename_axis(index="tmt_label")


class LOPITExperiment(Mapping[Gene, LOPITDataEntry]):
    def __init__(self, run: LOPITRun, name: str):
        self.run = run
        self.name = name
        self.nr_labels: int | None = None

        self._entries: dict[Gene, LOPITDataEntry] = {}

    def add_entry(self, entry: LOPITDataEntry):
        if entry.gene in self:
            raise ValueError("Gene already has an entry!")

        if self.nr_labels is None:
            self.nr_labels = len(entry.data)
        if len(entry.data) != self.nr_labels:
            raise ValueError("Data inconsistency - nr of labels is not the "
                             "expected number!")

        self._entries[entry.gene] = entry

    def __hash__(self):
        return hash((self.run, self.name))

    def __eq__(self, other: LOPITExperiment):
        if not isinstance(other, LOPITExperiment):
            return NotImplemented
        return hash(self) == hash(other)

    def __getitem__(self, gene: Gene):
        return self._entries[gene]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def to_dataframe(self):
        series = [entry.to_series() for entry in self._entries.values()]
        return pd.concat(
            {s.name: s for s in series},
            names=("gene_id", "experiment"),
        )


class LOPITExperimentCollection(Mapping[str, LOPITExperiment]):
    def __init__(self, experiments: list[LOPITExperiment]):
        self._experiments = {
            experiment.name: experiment for experiment in experiments
        }
        self.genes: dict[str, Gene] = {
            gene.gene_id: gene
            for experiment in self._experiments.values()
            for gene in experiment
        }

    def __getitem__(self, name: str):
        return self._experiments[name]

    def __iter__(self):
        return iter(self._experiments)

    def __len__(self):
        return len(self._experiments)

    def to_dataframe(self):
        return pd.concat(
            (exp.to_dataframe() for exp in self._experiments.values())
        )


class LOPITRun(LOPITExperimentCollection):
    def __init__(self, name: str):
        self.name = name
        super().__init__([])

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: LOPITRun):
        if not isinstance(other, LOPITRun):
            return NotImplemented
        return hash(self) == hash(other)

    def add_experiment(self, experiment: LOPITExperiment):
        if experiment.run != self:
            raise ValueError("Data inconsistency - experiment is not from "
                             "this run.")

        if experiment.name in self:
            raise ValueError("Experiment is already in run!")

        self._experiments[experiment.name] = experiment

    def get_prevailing_organism(self):
        organisms = Counter[str]()
        for gene in self.genes.values():
            organisms[gene.organism] += 1

        total = organisms.total()
        most_common, count = organisms.most_common(1)[0]
        fraction = count / total
        if fraction < 0.75:
            return None
        return OrthoMCL.get_organism_info(most_common)

    @staticmethod
    def from_dataframe(name: str, dataframe: pd.DataFrame):
        run = LOPITRun(name)
        for exp_name, exp_data in dataframe.groupby("experiment"):
            experiment = LOPITExperiment(run, exp_name)
            tmt_labels = list(exp_data.columns)
            for gene_id, row in exp_data.droplevel("experiment").iterrows():
                gene = Gene.from_gene_id(gene_id)
                run.genes[gene.gene_id] = gene
                data_entry = LOPITDataEntry(
                    experiment,
                    gene,
                    row.values,
                    tmt_labels
                )
                experiment.add_entry(data_entry)
                gene.add_entry(data_entry)

            run.add_experiment(experiment)

        return run
