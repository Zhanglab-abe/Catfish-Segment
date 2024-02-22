CONFIG_FILE="/home/agricoptics/Desktop/CatFish/mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k-1.py"
MODEL_FILE="/home/agricoptics/Desktop/CatFish/mmsegmentation/save/iter_7200.pth"
OUTPUT_FILE="/home/agricoptics/Desktop/CatFish/mmsegmentatation/result/"
python /home/agricoptics/Desktop/CatFish/mmsegmentation/tools/test.py \
    ${CONFIG_FILE} \
    ${MODEL_FILE}  \
    --out ${OUTPUT_FILE}/results.pkl \
    --eval mIoU \
