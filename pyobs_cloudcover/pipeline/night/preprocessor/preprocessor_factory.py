from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_options import PreprocessorOptions


class PreprocessorFactory:
    def __init__(self, options: PreprocessorOptions):
        self._options = options

    def __call__(self) -> Preprocessor:
        masker = ImageMasker.from_npy_file(self._options.mask_file_path) if self._options.mask_file_path else None
        return Preprocessor(self._options.bin_size, self._options.flip_vertical, masker)
