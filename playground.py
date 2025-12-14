from Pipeline.pipeline import Pipeline
# from Pipeline.pipeline_factory import ImageEncoderExamplePipelineFactory
from Pipeline.pipeline_factory import ImageColorizationPipelineFactory

# example pipeline for image encoding one image
# example: Pipeline = ImageEncoderExamplePipelineFactory.create()

# features, skips = example.run()

# print(features.shape)

pipeline: Pipeline = ImageColorizationPipelineFactory.create()
 
pred_rgb = pipeline.run()

print(pred_rgb)