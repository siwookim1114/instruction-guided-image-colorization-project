from abc import ABC, abstractmethod

from models.unet_encoder import UNetEncoder
from utils.download_img import Downloader
from utils.generate_instructions import build_instruction_json
from utils.preprocess import InstructionColorizationDataset


class Runner(ABC):
    @abstractmethod
    def run(self, inputs: dict = None): ...


class DownloaderRunner(Downloader, Runner):
    def run(self, inputs: dict = None):
        self.download()
        self.extract()


class InstructionRunner(Runner):
    def __init__(self, split_name, annot_path):
        self.split_name = split_name
        self.annot_path = annot_path

    def run(self, inputs: dict = None):
        build_instruction_json(self.annot_path, self.split_name)


class PreprocessorRunner(InstructionColorizationDataset, Runner):
    def run(self, inputs: dict = None):
        return self


class UNetEncoderRunner(UNetEncoder, Runner):
    def run(self, inputs: dict = None):
        self.eval()
        return self
