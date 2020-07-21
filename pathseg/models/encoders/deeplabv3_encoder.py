import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(encoder_weights=None)
print(model)
