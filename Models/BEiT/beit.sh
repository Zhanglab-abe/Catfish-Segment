
tools/dist_train.sh \
    /home/agricoptics/Desktop/CatFish/mmsegmentatation_beit/configs/beit/upernet_beit-base_640x640_160k_ade20k_ms.py 1 \
    --work-dir /home/agricoptics/Desktop/CatFish/mmsegmentatation_beit/checkpoints --seed 0  --deterministic \
    --options model.pretrained=https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth
