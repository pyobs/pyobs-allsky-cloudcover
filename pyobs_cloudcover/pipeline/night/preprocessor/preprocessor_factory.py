from pyobs_cloudcover.pipeline.night.preprocessor.background_remover import BackgroundRemover
from pyobs_cloudcover.pipeline.night.preprocessor.image_binner import ImageBinner
from pyobs_cloudcover.pipeline.night.preprocessor.image_masker import ImageMasker
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor import Preprocessor
from pyobs_cloudcover.pipeline.night.preprocessor.preprocessor_options import PreprocessorOptions


class PreprocessorFactory(object):
    def __init__(self, options: PreprocessorOptions):
        self._options = options

    def __call__(self) -> Preprocessor:
        if self._options.mask_file_path == "":
            mask = lambda x: x
        else:
            mask = ImageMasker.from_npy_file(self._options.mask_file_path)

        binner = ImageBinner(self._options.bin_size)
        background_remover = BackgroundRemover(self._options.bkg_sigma_clip, self._options.bkg_box_size)
        preprocessor = Preprocessor(mask, binner, background_remover)

        return preprocessor
