# python \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/demo/image_demo.py \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/test/Original_Data/Catfish2022-8-3/IMG_0510.HEIC.jpg \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/checkpoints_pspnet/iter_7200.pth --device cuda --out-file /home/agricoptics/Desktop/CatFish/mmsegmentation/result/topresultn.jpg

# python \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/demo/image_demo.py \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/test/Original_Data/Catfish2022-8-3/IMG_0513.HEIC.jpg \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py  \
#     /home/agricoptics/Desktop/CatFish/mmsegmentation/checkpoints_pspnet/iter_7200.pth --device cuda --out-file /home/agricoptics/Desktop/CatFish/mmsegmentation/result/sideresultn.jpg

python \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/demo/image_demo.py \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/test/data_raw/images_raw/cl8iz4sa205is08zk34d417fk.jpg \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/checkpoints_pspnet/iter_7200.pth --device cuda --out-file /home/agricoptics/Desktop/CatFish/mmsegmentation/result/topresultn22.jpg

python \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/demo/image_demo.py \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/data/ade/ADEChallengeData2016/test/data_raw/images_raw/cl8j0h1fl0ir408zk0nv18gpl.jpg \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_80k_ade20k.py \
    /home/agricoptics/Desktop/CatFish/mmsegmentation/checkpoints_pspnet/iter_7200.pth --device cuda --out-file /home/agricoptics/Desktop/CatFish/mmsegmentation/result/sideresult22.jpg
