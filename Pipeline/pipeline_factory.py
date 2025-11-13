from abc import ABC, abstractmethod
import os
from Pipeline.pipe import Pipe
from Pipeline.pipeline import Pipeline
from config import DATASETS, ANNOTATIONS
from Pipeline.runner import (
    Runner,
    DownloaderRunner,
    InstructionRunner,
    PreprocessorRunner,
    UNetEncoderRunner,
)
from transformers import AutoTokenizer


class PipelineFactory(ABC):
    @abstractmethod
    def create(): ...


class ExampleRunner(Runner):
    def run(self, inputs: dict = None):
        dataset = inputs["preprocess"]
        encoder = inputs["image_encoder"]

        if not encoder or not dataset:
            raise ValueError("Dependencies have resolved erroneously.")

        print(encoder)
        print(dataset)

        return encoder(dataset[0]["L"].unsqueeze(0))


class ImageEncoderExamplePipelineFactory(PipelineFactory):
    @classmethod
    def create(cls):
        pipeline = Pipeline()

        ## get downloader pipe
        downloader_pipes: list[Pipe] = []
        for ds in DATASETS:
            runner = cls._create_download_runner(ds["url"], ds["zip"], ds["dir"])
            pipe = Pipe(runner, ds["url"])
            downloader_pipes.append(pipe)

        ## get instruction pipe
        instruction_pipes: list[Pipe] = []
        for split_name, annot_path in ANNOTATIONS.items():
            runner = cls._create_instruction_runner(split_name, annot_path)
            pipe = Pipe(runner, split_name)
            for dl in downloader_pipes:
                pipe.add_dependency(dl)
            instruction_pipes.append(pipe)

        ## get preprocess pipe
        preprocess_pipe = Pipe(cls._create_preprocess_runner(), "preprocess")
        for ins in instruction_pipes:
            preprocess_pipe.add_dependency(ins)

        ## get image encoder pipe
        image_encoder_pipe = Pipe(cls._create_img_encoder_runner(), "image_encoder")

        ## Example pipe
        example_pipe = Pipe(ExampleRunner(), "example_output")
        example_pipe.add_dependency(preprocess_pipe).add_dependency(image_encoder_pipe)

        ## create pipeline
        pipeline.extend_pipe(downloader_pipes).extend_pipe(instruction_pipes)
        pipeline.add_pipe(preprocess_pipe).add_pipe(image_encoder_pipe).add_pipe(
            example_pipe
        )
        pipeline.set_output_pipe(example_pipe)

        return pipeline

    @classmethod
    def _create_download_runner(cls, url, dest_path, extract_path):
        return DownloaderRunner(url, dest_path, extract_path)

    @classmethod
    def _create_instruction_runner(cls, split_name, annot_path):
        return InstructionRunner(split_name, annot_path)

    @classmethod
    def _create_preprocess_runner(cls):
        return PreprocessorRunner(
            img_dir=os.path.abspath("data/raw/train2014/train2014"),
            ann_path=os.path.abspath("data/processed/instructions_train2014.json"),
            tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        )

    @classmethod
    def _create_img_encoder_runner(cls):
        return UNetEncoderRunner(1, 64)
