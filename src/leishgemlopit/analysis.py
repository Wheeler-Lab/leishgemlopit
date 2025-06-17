from collections import Counter, defaultdict
import pathlib

from .supervised import SupervisedTAGMCollection
from .tsne import TSNEAnalysis
from .unsupervised import UnsupervisedHDBSCAN
from .lopit import DEFAULT_TMT_LABELS, LOPITRun
from .tryptag import TrypTagMarkerFactory
from .markers import MarkerGenerator, MarkerFactory, Markers
from orthomcl import OrthoMCL

import pandas as pd

FIXTURE_PATH = pathlib.Path(__file__).parent / "__assets__"


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

    run = LOPITRun.from_dataframe(run_name, lopit_data)

    tsne = TSNEAnalysis(run)
    tsne.to_png(f"{run.name}_tsne.png")

    markers = Markers(markerfactory, list(run.genes), tsne)
    markercounter = Counter[str](
        [m for mset in markers.values() for m in mset])
    for m, c in markercounter.most_common():
        print(m, c)
    markers.to_png(f"{run.name}_markers.png")

    clusters = UnsupervisedHDBSCAN(run, tsne, min_samples=3, markers=markers)
    clusters.to_png(f"{run.name}_unsupervised_clusters.png")

    assigned = SupervisedTAGMCollection(
        run,
        markers,
        tsne_analysis=tsne,
    )
    assigned.to_png(f"{run.name}_supervised_clusters.png")


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("lopit_data", type=pathlib.Path)

    args = parser.parse_args()

    df = pd.read_csv(args.lopit_data).set_index("geneid").iloc[:, :44]
    df.columns = pd.MultiIndex.from_product(
        [(1, 2, 3, 4), DEFAULT_TMT_LABELS],
        names=["experiment", "tmt_labels"]
    )
    df = df[df.notna().all(axis=1)]
    df: pd.DataFrame = df.stack("experiment", future_stack=True)

    analysis(args.name, df)
