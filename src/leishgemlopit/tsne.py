import io
from typing import Mapping, NamedTuple

from leishgemlopit.lopit import LOPITExperimentCollection

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from leishgemlopit.utils import PNGMixin


class TSNEPoint(NamedTuple):
    x: float
    y: float


class TSNEAnalysis(Mapping[str, TSNEPoint], PNGMixin):
    def __init__(self, lopit_experiments: LOPITExperimentCollection):
        data = lopit_experiments.to_dataframe()
        self._do_tsne(data)

    def _do_tsne(self, data: pd.Series):
        df = data.unstack(["experiment", "tmt_label"])
        result = pd.DataFrame(
            TSNE(perplexity=30, max_iter=10000).fit_transform(df),
            index=df.index,
        )
        self._data: dict[str, TSNEPoint] = {
            str(geneid): TSNEPoint(*row) for geneid, row in result.iterrows()
        }

    def __getitem__(self, geneid: str):
        return self._data[geneid]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dataframe(self, include_tsne=True):
        return pd.DataFrame(
            [
                {
                    "geneid": geneid,
                    "x": p.x,
                    "y": p.y
                }
                for geneid, p in self.items()
            ]
        ).set_index("geneid")

    def _figure(self):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        df = self.to_dataframe()
        ax.scatter(
            df.x, df.y,
            s=5,
            marker=MarkerStyle("o", fillstyle="none"),
            linewidth=0.5,
            color="black",
        )

        ax.axis("off")

        return fig

    def _repr_png_(self):
        figure = self._figure()
        png_bytes = io.BytesIO()
        with png_bytes:
            figure.savefig(png_bytes, format="png")
            ret_value = png_bytes.getvalue()
        plt.close(figure)
        return ret_value
