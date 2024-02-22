_base_ = [
    '../_base_/models/upernet_beit_theva.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

#My code Starts here
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class LitModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = ...
        self.log("val_loss", loss)

#My code ends here

model = dict(
    pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)))

#My code Starts here
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
#My code ends here

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

trainer.fit(model)
