from collections import defaultdict
from collections.abc import Mapping
import argparse
import pathlib
import re
from typing import NamedTuple
from zipfile import ZipFile

from .constants import FIXTURE_PATH
from .supervised import SupervisedTAGMCollection
from .tsne import TSNEAnalysis, TSNEPoint
from .unsupervised import UnsupervisedHDBSCAN
from .lopit import DEFAULT_TMT_LABELS, Gene, LOPITRun
from .tryptag import TrypTagMarkerFactory
from .markers import MarkerGenerator, MarkerFactory, Markers
from orthomcl import OrthoMCL

import pandas as pd


class TBruceiMarkerTermFile(MarkerGenerator):
    def __init__(self, term: str, filename: pathlib.Path):
        self.term = term
        with filename.open("r") as f:
            self.gene_ids = {
                gene_id.strip() for gene_id in f
            }

    def __call__(self, gene_id):
        treu927_ids = {
            entry.gene_id for entry in OrthoMCL[gene_id]["tbrt"]
        }
        if not treu927_ids.isdisjoint(self.gene_ids):
            return {self.term}
        return set()


class LmexMultiSheetExcelMarkerTermFile(MarkerGenerator):
    def __init__(self, filename: pathlib.Path):
        terms = defaultdict(set)

        additional_markers = pd.read_csv(filename).set_index("geneid").term

        for gene_id, term in additional_markers.items():
            terms[gene_id].add(term)

        self.terms = dict(terms)

    def __call__(self, gene_id: str):
        lmex_ids = {
            entry.gene_id for entry in OrthoMCL[gene_id]["lmex"]
        }
        terms: set[str] = set()
        for lmex_id in lmex_ids:
            if lmex_id in self.terms:
                terms |= self.terms[lmex_id]
        return terms


def DEFAULT_MARKER_FACTORY():
    return MarkerFactory([
        TrypTagMarkerFactory(),
        TBruceiMarkerTermFile(
            "ribosome", FIXTURE_PATH / "brucei.ribo.txt"),
        TBruceiMarkerTermFile(
            "mitochondrial ribosome", FIXTURE_PATH / "brucei.mitoribo.txt"),
        LmexMultiSheetExcelMarkerTermFile(
            FIXTURE_PATH / "leishmania_mexicana_additional_markers.csv",
        )
    ])


class LOPITAnalysisResult(NamedTuple):
    gene_id: Gene
    gene_description: str | None
    tsne_xy: TSNEPoint
    marker: str | None
    unsupervised_cluster: int
    unsupervised_label: str | None
    tagm_label: str
    tagm_confidence: float


class Analysis(Mapping[str, LOPITAnalysisResult]):
    def __init__(
        self,
        run: LOPITRun,
        marker_factory=DEFAULT_MARKER_FACTORY
    ):
        self.run = run
        self.prevailing_organism = self.run.get_prevailing_organism()
        self.tsne = TSNEAnalysis(run)
        self.markers = Markers(marker_factory(), list(run.genes), self.tsne)
        self.analyses = [self.tsne, self.markers]

        self.clusters = UnsupervisedHDBSCAN(
            run,
            self.markers,
            tsne_analysis=self.tsne,
            min_samples=3,
        )
        self.analyses.append(self.clusters)

        self.assigned = SupervisedTAGMCollection(
            run,
            self.markers,
            tsne_analysis=self.tsne,
        )
        self.analyses.append(self.assigned)

    @staticmethod
    def from_csv(
        file: pathlib.Path,
        marker_factory=DEFAULT_MARKER_FACTORY,
    ):
        file = pathlib.Path(file)

        df = pd.read_csv(file)
        tmt_re = "|".join(DEFAULT_TMT_LABELS)
        columns_re = re.compile(
            rf"Abundances \(Grouped\): (?P<experiment>[A-Z]), (?P<tmt>{tmt_re})$")  # noqa: E501
        relevant_columns = {"UniProt Entry Name": "geneid"}
        for column in df.columns:
            if m := columns_re.match(column):
                relevant_columns[column] = (m["experiment"], m["tmt"])
        df = df.loc[:, list(relevant_columns)]
        df.columns = list(relevant_columns.values())
        df.loc[:, "geneid"] = df.geneid.str.replace("-t42_1-p1", "")
        df = df.set_index("geneid").fillna(0.0)
        df.columns = pd.MultiIndex.from_tuples(
            df.columns.tolist(), names=("experiment", "tmt_label"))
        df = df.stack("experiment", future_stack=True)
        df: pd.DataFrame = (df.T / df.sum(axis=1)).T
        df = df.unstack("experiment")
        df = df[df.notna().all(axis=1)].stack("experiment", future_stack=True)

        run = LOPITRun.from_dataframe(file.stem, df)
        return Analysis(run, marker_factory)

    def __getitem__(self, geneid: str):
        try:
            gene = self.run.genes[geneid]
        except KeyError:
            raise KeyError(f"Gene {geneid} not found.")
        gene_description = gene.description

        tsne_xy = self.tsne[geneid]
        marker = self.markers[geneid]
        unsupervised_cluster = self.clusters[geneid]
        if self.clusters.was_assigned(geneid):
            unsupervised_label = (
                self.clusters.get_presentation_label(unsupervised_cluster))
        else:
            unsupervised_label = None
        tagm_label = self.assigned[geneid]
        tagm_confidence = self.assigned.get_confidence(geneid)

        return LOPITAnalysisResult(
            geneid,
            gene_description,
            tsne_xy,
            marker,
            unsupervised_cluster,
            unsupervised_label,
            tagm_label,
            tagm_confidence,
        )

    def __iter__(self):
        return iter(self.run.genes)

    def __len__(self):
        return len(self.run.genes)

    def to_dataframe(self):
        return pd.DataFrame.from_records(
            list(self.values()),
            columns=LOPITAnalysisResult._fields,
        ).set_index("gene_id")

    def save(self, path: pathlib.Path = pathlib.Path(".")):
        path = pathlib.Path(path)
        run_name = self.run.name

        with ZipFile(path / (run_name + ".zip"), "w") as zip:
            self.tsne.to_png(zip.open(f"{run_name}_tsne.png", "w"))
            self.markers.to_png(zip.open(f"{run_name}_markers.png", "w"))
            self.clusters.to_png(
                zip.open(f"{run_name}_unsupervised_clusters.png", "w"))
            self.assigned.to_png(
                zip.open(f"{run_name}_supervised_clusters.png", "w"))
            with zip.open(f"{run_name}_results.xlsx", "w") as excel_zip:
                with pd.ExcelWriter(excel_zip) as excel:
                    self.to_dataframe().to_excel(excel, sheet_name="Results")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("lopit_data", type=pathlib.Path)
    args = parser.parse_args()
    analysis = Analysis.from_csv(args.lopit_data)
    analysis.save()
