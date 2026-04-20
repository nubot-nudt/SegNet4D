### Environments: pytorch 2.4.1, cuda 11.8, python 3.8

### Tested on ubuntu20.04, 4090GPU

#### 1. create env:

```bash
conda create -n segnet4d python=3.8
```

#### 2. install torch

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

#### 3. install sptr library

```bash
cd models/SparseTransformer
python3 setup.py install
```

If you encounter an error regarding `THC.h,` please replace `<THC/THC.h>` with `<ATen/ATen.h>`.

#### 4. install spconv-cu118

```bash
pip install spconv-cu118
```

#### 5. install pytorch-lightnling

```bash
pip install pytorch-lightning==1.5.10
```

#### 6. install other library

```bash
pip install torch-scatter==2.0.9
pip install torch-geometric==1.7.2
pip install torch-cluster==1.6.1
pip install torch-sparse==0.6.18
pip install timm==0.9.7
```

It should take a long time to install.

#### 7. install array_index.cpp

```bash
cd utils/src
mkdir build
cd build
cmake ..
make -j8
```

Put the `Array_Index.cpython-38-x86_64-linux-gnu.so` file into `SegNet4D/dataloader/src`

#### 8. run segnet4d

```bash
python scripts/predict_semantickitti.py --cfg_file config/semantickitti/semantickitti_config.yaml --data_path /home/luankai/Test_wangneng/data/kitti/sequences/ --ckpt ./semantickitti.ckpt
```

