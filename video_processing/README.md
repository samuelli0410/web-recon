## `video_to_image_sequence.py`

Slice a video file into multiple image files.

```
python video_to_image_sequence.py --src_file ./test_video.mp4 --dst_dir ./sliced_images
```

### Arguments

* `src_file`: Path to the source video file.
* `dst_dir`: Target directory to save image files. Defaults to `None` - will be saved in the same directory as source file.
* `n_frames`: Number of frames such that the video is sampled every `n_frames` frames. Defaults to `5`.


## `point_cloud_generation.py`

Transform a video file into Point Cloud file (`.pcd`).
```
python point_cloud_generation.py --src_file ./test_video.mp4
```

### Arguments

* `src_file`: Path to the source video file.
* `dst_dir`: Target directory to save image files. Defaults to `None` - will be saved in the same directory as source file.
* `depth_scale`: Defaults to `1`.


## `graph_generation.py`
