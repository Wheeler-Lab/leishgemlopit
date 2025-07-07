from collections import Counter, defaultdict
import argparse
import pathlib
import re
from zipfile import ZipFile

from .constants import FIXTURE_PATH
from .supervised import SupervisedTAGMCollection
from .tsne import TSNEAnalysis
from .unsupervised import UnsupervisedHDBSCAN
from .lopit import DEFAULT_TMT_LABELS, LOPITRun
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


def analysis(run_name: str, lopit_data: pd.DataFrame):
    run = LOPITRun.from_dataframe(run_name, lopit_data)

    markerfactory = MarkerFactory([
        TrypTagMarkerFactory(),
        TBruceiMarkerTermFile(
            "ribosome", FIXTURE_PATH / "brucei.ribo.txt"),
        TBruceiMarkerTermFile(
            "mitochondrial ribosome", FIXTURE_PATH / "brucei.mitoribo.txt"),
        LmexMultiSheetExcelMarkerTermFile(
            FIXTURE_PATH / "leishmania_mexicana_additional_markers.csv",
        )
    ])

    descriptions = pd.Series(
        {
            gene.gene_id: gene.description if gene.description != "unknown" else ""
            for gene in {
                gene for experiment in run.values()
                for gene in experiment
            }
        },
        name="description",
    )

    with ZipFile(run_name + ".zip", "w") as zip:
        tsne = TSNEAnalysis(run)
        tsne.to_png(zip.open(f"{run.name}_tsne.png", "w"))

        markers = Markers(markerfactory, list(run.genes), tsne)
        markers.to_png(zip.open(f"{run.name}_markers.png", "w"))

        clusters = UnsupervisedHDBSCAN(run, markers, tsne, min_samples=3)
        clusters.to_png(zip.open(f"{run.name}_unsupervised_clusters.png", "w"))

        assigned = SupervisedTAGMCollection(
            run,
            markers,
            tsne_analysis=tsne,
        )
        assigned.to_png(zip.open(f"{run.name}_supervised_clusters.png", "w"))

        tables = {
            "Gene": descriptions,
        }
        tables.update(
            {
                key: component.to_dataframe(include_tsne=False)
                for key, component in [
                    ("t-SNE", tsne),
                    ("Markers", markers),
                    ("Unsupervised clustering", clusters),
                    ("TAGM-MAP", assigned),
                ]
            }
        )
        table = pd.concat(
            tables,
            axis=1,
        )
        table = table.sort_index().fillna("")
        print(table.loc[:, ("Unsupervised clustering", "cluster")])
        print(table.loc[:, ("Gene", "description")])
        print(table.loc[:, ("Markers", "marker")])

        nr_of_unknown_unsupervised = (
            (
                (table.loc[:, ("Unsupervised clustering", "cluster")] >= 0) &
                (table.loc[:, ("Gene", "description")] == "") &
                (table.loc[:, ("Markers", "marker")] == "")
            ).sum()
        )
        nr_of_unknown_tagm = (
            (
                (table.loc[:, ("TAGM-MAP", "label")] != "") &
                (table.loc[:, ("Gene", "description")] == "") &
                (table.loc[:, ("Markers", "marker")] == "")
            ).sum()
        )
        summary = pd.DataFrame(
            [
                {
                    "Analysis": "Unsupervised clustering",
                    "Number of unknown genes assigned": nr_of_unknown_unsupervised,
                },
                {
                    "Analysis": "TAGM-MAP",
                    "Number of unknown genes assigned": nr_of_unknown_tagm,
                },
            ]
        ).set_index("Analysis")

        markercounter = Counter[str](
            [m for mset in markers.values() for m in mset])
        markersummary = pd.Series(
            {
                marker: count
                for marker, count in markercounter.most_common()
            },
            name="Count",
        ).to_frame()

        with zip.open(f"{run.name}_results.xlsx", "w") as excel_zip:
            with pd.ExcelWriter(excel_zip) as excel:
                markersummary.to_excel(excel, sheet_name="Markers summary")
                summary.to_excel(excel, sheet_name="Analysis summary")
                table.to_excel(excel, sheet_name="Assignments")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("lopit_data", type=pathlib.Path)

    args = parser.parse_args()

    df = pd.read_csv(args.lopit_data) #.set_index("geneid").iloc[:, :44]
    tmt_re = "|".join(DEFAULT_TMT_LABELS)
    columns_re = re.compile(rf"Abundances \(Grouped\): (?P<experiment>[A-Z]), (?P<tmt>{tmt_re})$")
    relevant_columns = {"UniProt Entry Name": "geneid"}
    for column in df.columns:
        if m := columns_re.match(column):
            relevant_columns[column] = (m["experiment"], m["tmt"])
    df = df.loc[:, list(relevant_columns)]
    df.columns = list(relevant_columns.values())
    df.loc[:, "geneid"] = df.geneid.str.replace("-t42_1-p1", "")
    df = df.set_index("geneid").fillna(0.0)
    df.columns = pd.MultiIndex.from_tuples(df.columns.tolist(), names=("experiment", "tmt_label"))
    df = df.stack("experiment", future_stack=True)
    df: pd.DataFrame = (df.T / df.sum(axis=1)).T
    df = df.unstack("experiment")
    df = df[df.notna().all(axis=1)].stack("experiment", future_stack=True)

    analysis(args.name, df)
