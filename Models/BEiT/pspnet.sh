 tools/dist_train.sh \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py  1 \
    --work-dir /home/agricoptics/Desktop/CatFish/mmsegmentation/checkpoints_pspnet --seed 0  --deterministic \
    --options model.pretrained=https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x512_80k_ade20k/pspnet_r101-d8_512x512_80k_ade20k_20200614_031423-b6e782f0.pth
