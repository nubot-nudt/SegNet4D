# nuScenes multi-scan semantic labels generation

This  generates nuscenes multi-scan semantic labels based on the SemanticKITTI format.

## Label Mapping

```yaml
  # the raw nuscenes static label
  0: 'noise'
  1: 'barrier'
  2: 'bicycle'
  3: 'bus'
  4: 'car'
  5: 'construction_vehicle'
  6: 'motorcycle'
  7: 'pedestrian'
  8: 'traffic_cone'
  9: 'trailer'
  10: 'truck'
  11: 'driveable_surface'
  12: 'other_flat'
  13: 'sidewalk'
  14: 'terrain'
  15: 'manmade'
  16: 'vegetation'
  # new generating moving class
  17: 'moving_car'
  18: 'moving_bus'
  19: 'moving_truck'
  20: 'moving_construction'
  21: 'moving_trailer'
  22: 'moving_motorcyclist'
  23: 'moving_cyclist'
  24: 'moving_pedestrian'
```

## Usage

**1.Raw dataset**

You can download the raw nuscenes dataset on [official website](https://www.nuscenes.org/)

**2.run nuscenes_process.py**

```
python nuscenes_process.py
```

And then, some files will be generated in the `./data/nuScenes_kitti` directory

<details>
    <summary><strong>Data structure</strong></summary>

```yaml
└── nuScenes_kitti
  ├── train/           
  │   ├── 0001
  |   |	├── boundingbox  # bounding box label for keyframe
  |   | | ├── 000000.npy
  |   | | ├── 000001.npy
  |   | | └── ...
  |   |	├── labels   # multi-scan label for keyframe (class 0-24)
  |   | | ├── 000000.label
  |   | | ├── 000001.label
  |   |	| └── ...
  |   |	├── sem_labels   # single-scan label for keyframe (class 0-16)
  |   | | ├── 000000.label
  |   | | ├── 000001.label
  |   |	| └── ...
  |   |	├── velodyne   # LiDAR scan for keyframe 
  |   | | ├── 000000.bin
  |   | | ├── 000001.bin
  |   |	| └── ...
  |   |	├── poses.txt # poses for keyframe 
  |   ├── 0002
  |	  |	  ├── boundingbox
  |	  |	  ├── labels
  |	  |	  ├── sem_labels
  |	  |	  ├── velodyne
  |	  |	  ├── poses.txt
  |	  └── ...
  └──  val
    ├── 0003/ # for validation
  	└── ...
```

</details>  

## Visual

Run the following command to visualize the results of multi-scan semantic labels and mos labels.

Press key  `n`  to show next frame

Press key  `b`  to show last frame

Press key  `q`  to quit display

**visual of semantic label**

```bash
cd visual
python vis_sem.py
```

**visual of mos label**

```bash
cd visual
python vis_mos.py
```

**visual of bounding box label**

```bash
cd visual
python vis_box.py
```



## Contact

Any question or suggestions are welcome!

Neng Wang: nwang@nudt.edu.cn and Xieyuanli Chen: xieyuanli.chen@nudt.edu.cn

## Acknowledgment

We thank for the opensource codebases, [MapMOS](https://github.com/PRBonn/MapMOS.git), [AutoMOS](https://github.com/PRBonn/auto-mos.git)

## License

This project is free software made available under the MIT License. For details see the LICENSE file.

