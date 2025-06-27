from collections import defaultdict
import json
import pathlib

import numpy as np

from tryptag import CellLine, GeneNotFoundError, TrypTag
from tryptag.datasource import CellLineStatus

from leishgemlopit.constants import ANNOTATIONS_BACKGROUND_LIKE
from leishgemlopit.markers import MarkerGenerator

from orthomcl import OrthoMCL

FIXTURE_PATH = pathlib.Path(__file__).parent / "__assets__"

SIGNAL_THRESHOLD_QUANTILE = 0.25

EXCLUDE_MODIFIERS = {"weak", "<10%", "review", "unconvincing"}

_ANNOTATION_REMAPPING = {
    ('cytoplasm', 'nuclear lumen', 'flagellar cytoplasm'): 'cytoplasm',
    ('cytoplasm', 'nucleoplasm'): "nuccyto",
    ('cytoplasm',): 'cytoplasm',
    ('nucleoplasm',): 'nucleoplasm',
    ('nucleolus',): 'nucleolus',
    ('nuclear envelope',): 'nuclear envelope',
    ('nuclear pore',): 'nuclear pore',
    ('basal body',): 'basal body',
    ('axoneme',): 'axoneme',
    ('flagellar tip',): 'axoneme',
    ('paraflagellar rod',): 'paraflagellar rod',
    ('kinetoplast', 'mitochondrion'): 'matrix',
    ('antipodal sites', 'mitochondrion'): 'matrix',
    ('mitochondrion',): 'intermembrane',
    ('glycosome',): 'glycosome',
    ('acidocalcisome',): 'acidocalcisome',
    ('lipid droplets',): 'lipid droplets',
    ('Golgi apparatus',): 'Golgi apparatus',
    ('endocytic',): 'endocytic',
    ('endoplasmic reticulum',): 'endoplasmic reticulum',
    ('flagellar pocket collar',): 'flagellar pocket cytoskeleton',
    ('microtubule quartett',): 'flagellar pocket cytoskeleton',
    ('hook complex',): 'flagellar pocket cytoskeleton',
    ('flagellum attachment zone',): 'flagellar pocket cytoskeleton',
    ('flagellar pocket membrane',): 'flagellar pocket',
    ('flagellar pocket',): 'flagellar pocket',
    ('intraflagellar transport particle',): 'intraflagellar transport particle',  # noqa: E501
    ('cortical cytoskeleton',): 'cortical cytoskeleton',
    ('cleavage furrow',): 'cortical cytoskeleton',
}
ANNOTATION_REMAPPING = {
    frozenset(k): v for k, v in _ANNOTATION_REMAPPING.items()}


IGNORE_PER_TERMINUS = {
    "N": {
        "endoplasmic reticulum",
        "mitochondrion",
        "nuclear envelope",
        "kinetoplast",
        "antipodal sites"
    },
    "C": {"glycosomes"}
}


class TrypTagMarkerFactory(MarkerGenerator):
    def __init__(self):
        self.tryptag = TrypTag(dataset_name="procyclic")
        self._load_signals()

    def _load_signals(self):
        with (
            FIXTURE_PATH / "trypTag_signal_quantification.json"
        ).open("r") as f:
            self.signals = json.load(f)

        per_term = defaultdict(list)
        for gene_id, termini in self.signals.items():
            for terminus, data in termini.items():
                cell_line = self.tryptag.gene_list[gene_id][terminus]
                for localisation in (
                    cell_line.localisation.filter_by_modifiers(
                        exclude_modifiers=EXCLUDE_MODIFIERS)
                ):
                    if data["mean_signal"] is not None:
                        per_term[localisation.term].append(data["mean_signal"])
        self.signal_thresholds: dict[str, float] = {
            term: np.quantile(values, SIGNAL_THRESHOLD_QUANTILE)
            for term, values in per_term.items()
        }

    def get_tryptag_localisations(self, gene_id: str):
        treu927_entries = OrthoMCL[gene_id]["tbrt"]
        localisation_list: set[str] = set()
        for treu927_entry in treu927_entries:
            try:
                tt_gene = self.tryptag.gene_list[treu927_entry.gene_id]
            except GeneNotFoundError:
                continue

            cell_line: CellLine
            for cell_line in tt_gene.values():
                if (
                    cell_line.status != CellLineStatus.GENERATED or
                    cell_line.classified_faint == "y"
                ):
                    continue
                try:
                    signal = (
                        self.signals[cell_line.gene_id][cell_line.terminus])
                    mean_signal = signal["mean_signal"]
                    if mean_signal is None:
                        mean_signal = np.inf
                    nuccyto_ratio = signal["nuclear_to_cytoplasm_signal_ratio"]
                    if nuccyto_ratio is None:
                        nuccyto_ratio = np.inf
                except KeyError:
                    mean_signal = np.inf
                    nuccyto_ratio = None

                localisations: set[str] = {
                    a.term for a in
                    cell_line.localisation.filter_by_modifiers(
                        exclude_modifiers=EXCLUDE_MODIFIERS
                    )
                    if mean_signal > self.signal_thresholds[a.term]
                }
                localisations -= IGNORE_PER_TERMINUS[cell_line.terminus]
                localisations = self.remap_annotations(localisations)
                if "nuccyto" in localisations:
                    localisations.discard("nuccyto")
                    if nuccyto_ratio > 1.3:
                        localisations.add("nucleoplasm")
                    else:
                        localisations.add("cytoplasm")
                localisation_list = localisation_list | localisations
        return self.remove_background_like(localisation_list)

    def remap_annotations(self, localisations: set[str]):
        new_terms: set[str] = set()
        for matching_terms, replacement_term in ANNOTATION_REMAPPING.items():
            if localisations.issuperset(matching_terms):
                localisations -= matching_terms
                new_terms.add(replacement_term)
        return new_terms

    def remove_background_like(self, localisations: set[str]):
        if bg_removed := localisations.difference(ANNOTATIONS_BACKGROUND_LIKE):
            return bg_removed
        return localisations

    def __call__(self, gene_id: str):
        return self.get_tryptag_localisations(gene_id)
