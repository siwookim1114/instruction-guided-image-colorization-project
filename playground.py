from Pipeline.pipeline import Pipeline
from Pipeline.pipeline_factory import ImageEncoderExamplePipelineFactory

# example pipeline for image encoding one image
example: Pipeline = ImageEncoderExamplePipelineFactory.create()

features, skips = example.run()

print(features.shape)
