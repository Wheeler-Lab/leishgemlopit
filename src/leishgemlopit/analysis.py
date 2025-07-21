from collections import defaultdict
from collections.abc import Mapping
import argparse
from dataclasses import dataclass
import pathlib
from zipfile import ZipFile

from .constants import FIXTURE_PATH
from .supervised import SupervisedTAGMCollection, TAGMResultCollection
from .tsne import TSNEAnalysis, TSNEPoint
from .unsupervised import UnsupervisedHDBSCAN
from .lopit import Gene, LOPITRun
from .tryptag import TrypTagMarkerFactory
from .markers import (
    MarkerGenerator,
    MarkerCollection,
    Marker,
    MarkerConfidence,
    Markers,
    MarkerFactory,
)

import pandas as pd


class SingleMarkerTermFile(MarkerGenerator):
    def __init__(
        self,
        term: str,
        filename: pathlib.Path,
        source: str,
    ):
        self.term = term
        with filename.open("r") as f:
            self.gene_ids = {
                gene_id.strip() for gene_id in f
            }
        self.source = source

    def __call__(self, gene_id):
        ortholog_ids = self.all_orthologs_of(gene_id)
        if not ortholog_ids.isdisjoint(self.gene_ids):
            return MarkerCollection(
                gene_id,
                [
                    Marker(self.term, self.source, MarkerConfidence.EXCELLENT)
                ],
            )
        return MarkerCollection(gene_id)


class MultiMarkerTermFile(MarkerGenerator):
    def __init__(self, filename: pathlib.Path, source: str):
        terms = defaultdict(set)

        additional_markers = pd.read_csv(filename).set_index("geneid").term

        for gene_id, term in additional_markers.items():
            terms[gene_id].add(term)

        self.terms = dict(terms)

        self.source = source

    def __call__(self, gene_id: str):
        ortholog_ids = self.all_orthologs_of(gene_id)
        markers = MarkerCollection(gene_id)
        for ortholog_id in ortholog_ids:
            if ortholog_id in self.terms:
                for term in self.terms[ortholog_id]:
                    markers.add_marker(
                        Marker(
                            term,
                            self.source,
                            MarkerConfidence.EXCELLENT,
                        )
                    )
        return markers


def DEFAULT_MARKER_FACTORY():
    return MarkerFactory([
        TrypTagMarkerFactory(),
        SingleMarkerTermFile(
            "ribosome", FIXTURE_PATH / "brucei.ribo.txt", "T. brucei ribosome"),
        SingleMarkerTermFile(
            "mitochondrial ribosome",
            FIXTURE_PATH / "brucei.mitoribo.txt",
            "T. brucei mitochondrial ribosome"
        ),
        MultiMarkerTermFile(
            FIXTURE_PATH / "leishmania_mexicana_additional_markers.csv",
            "Eden's L. mexicana list",
        ),
        MultiMarkerTermFile(
            FIXTURE_PATH / "trypanosoma_cruzi_additional_markers.csv",
            "Eden's T. cruzi list",
        )
    ])


@dataclass
class LOPITAnalysisResult:
    gene_id: Gene
    gene_description: str | None
    tsne_xy: TSNEPoint
    marker: str | None
    unsupervised_cluster: int
    unsupervised_label: str | None
    tagm_results: TAGMResultCollection

    def to_row(self):
        tagm_label, tagm_confidence = self.tagm_results.assign()
        row = {
            "gene_id": self.gene_id,
            "gene_description": self.gene_description,
            "tsne_x": self.tsne_xy.x,
            "tsne_y": self.tsne_xy.y,
            "marker": self.marker,
            "unsupervised_cluster": self.unsupervised_cluster,
            "unsupervised_label": self.unsupervised_label,
            "tagm_label": tagm_label,
            "tagm_conf": tagm_confidence,
        }
        for experiment_name, result in self.tagm_results.items():
            row[f"tagm_{experiment_name}_label"] = result.label
            row[f"tagm_{experiment_name}_probability"] = result.probability
        return row


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
        tagm_results = self.assigned.get_tagm_results(geneid)

        return LOPITAnalysisResult(
            geneid,
            gene_description,
            tsne_xy,
            marker,
            unsupervised_cluster,
            unsupervised_label,
            tagm_results,
        )

    def __iter__(self):
        return iter(self.run.genes)

    def __len__(self):
        return len(self.run.genes)

    def to_dataframe(self):
        return pd.DataFrame([
            entry.to_row()
            for entry in self.values()
        ]).set_index("gene_id")

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
            with (
                zip.open(f"{run_name}_marker_sources.xlsx", "w")
                as markers_zip
            ):
                with pd.ExcelWriter(markers_zip) as excel:
                    df = self.markers.to_dataframe()
                    df.to_excel(excel, sheet_name="Markers")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("lopit_data", type=pathlib.Path)
    args = parser.parse_args()
    run = LOPITRun.from_csv(args.lopit_data)
    analysis = Analysis(run)
    analysis.save()
