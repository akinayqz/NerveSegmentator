"""Errors a user can act on. The command line tools print them without a traceback."""


class NerveSegmentatorError(Exception):
    pass


class InputError(NerveSegmentatorError, ValueError):
    """An argument or an input file is not usable."""


class ModelNotInstalled(NerveSegmentatorError):
    """The weights of a model are not installed and cannot be downloaded."""


class SpineSegmentationError(NerveSegmentatorError):
    """TotalSpineSeg is needed and cannot be run."""


class CoordinateError(NerveSegmentatorError):
    """The coordinate channels cannot be built from a spine segmentation.
    `report` is the QC report with the reasons in report['errors']."""

    def __init__(self, report):
        super().__init__("; ".join(report["errors"]))
        self.report = report
