from abc import ABC, abstractmethod
import os
from Pipeline.pipe import Pipe
from Pipeline.pipeline import Pipeline
from config import DATASETS, ANNOTATIONS
from Pipeline.runner import *
from transformers import AutoTokenizer

class PipelineFactory(ABC):
    @abstractmethod
    def create(): ...

class ImageColorizationPipelineFactory(PipelineFactory):
    @classmethod
    def create(cls):
        pipeline = Pipeline()
        # Get Device
        get_device_runner = GetDeviceRunner()
        get_device_pipe = Pipe(get_device_runner, "GetDevice")
        
        # Get Tokenizer
        get_tokenizer_runner = GetTokenizerRunner()
        get_tokenizer_pipe = Pipe(get_tokenizer_runner, "GetTokenizer")
        
        # Get Inputs
        get_input_runner = GetInputRunner()
        get_input_pipe = Pipe(get_input_runner, "GetInput")
        
        # Get Model
        get_model_runner = GetModelRunner()
        get_model_pipe = Pipe(get_model_runner, "GetModel")
        get_model_pipe.add_dependency(get_device_pipe)
        get_model_pipe.add_dependency(get_input_pipe)
        
        # Get TestSingleImage
        test_single_image_runner = TestSingleImageRunner()
        test_single_image_pipe = Pipe(test_single_image_runner, "TestSingleImage")
        test_single_image_pipe.add_dependency(get_device_pipe)
        test_single_image_pipe.add_dependency(get_input_pipe)
        test_single_image_pipe.add_dependency(get_tokenizer_pipe)
        test_single_image_pipe.add_dependency(get_model_pipe)
        
        # Build Pipeline 
        pipeline.add_pipe(get_device_pipe).add_pipe(get_tokenizer_pipe).add_pipe(get_input_pipe).add_pipe(get_model_pipe).add_pipe(test_single_image_pipe)
        pipeline.set_output_pipe(test_single_image_pipe)
        return pipeline

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


def lab_to_rgb__(L, ab):
    """
    L: (1, H, W) tensor in [-1, 1]
    ab: (2, H, W) tensor in [-1, 1]
    returns: RGB image uint8
    """
    if isinstance(L, torch.Tensor):
        L = L.detach().cpu()
    if isinstance(ab, torch.Tensor):
        ab = ab.detach().cpu()

    # Reshape if needed
    if len(L.shape) == 3:
        L = L.squeeze(0)  # (H, W)
    if len(ab.shape) == 3 and ab.shape[0] == 2:
        ab = np.transpose(ab, (1, 2, 0))  # (H, W, 2)

    # Undo normalization
    L = (L + 1.0) * 50.0
    ab = ab * 128.0

    # Stacking to LAB image
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype = np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab.transpose(1, 2, 0)

    # Convert LAB -> RGB
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Clip to valid range
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)

    return np.clip(rgb, 0, 1)
