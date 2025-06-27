import pathlib
import io


class PNGMixin:
    def to_png(self, output_file: pathlib.Path | str | io.FileIO):
        if (
            isinstance(output_file, pathlib.Path) or
            isinstance(output_file, str)
        ):
            output_file = pathlib.Path(output_file).open("wb")
        with output_file:
            output_file.write(self._repr_png_())

    def _repr_png_(self):
        raise NotImplementedError
