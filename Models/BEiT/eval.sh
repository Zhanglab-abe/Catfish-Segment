CONFIG_FILE="/home/agricoptics/Desktop/CatFish/mmsegmentatation_beit/configs/beit/upernet_beit-base_640x640_160k_ade20k_ms.py"
MODEL_FILE="/home/agricoptics/Desktop/CatFish/mmsegmentatation_beit/save/iter_7200.pth"
OUTPUT_FILE="/home/agricoptics/Desktop/CatFish/mmsegmentatation_beit/result/"
python /home/agricoptics/Desktop/CatFish/mmsegmentatation_beit/tools/test.py \
    ${CONFIG_FILE} \
    ${MODEL_FILE}  \
    --out ${OUTPUT_FILE}/results.pkl \
    --eval mIoU \
