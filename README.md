# Patient-specific reconstruction of volumetric computed tomography images from few-view projections via deep learning

## Implementing it with retrainer script.

NOTE! This is still in progress. Cloning and running this model won't work at this moment. 

# Error produced:

```
Traceback (most recent call last):
  File "/Users/georgeka/Desktop/uni/tooth project internship/xray-to-cbct/train.py", line 61, in <module>
    main()
  File "/Users/georgeka/Desktop/uni/tooth project internship/xray-to-cbct/train.py", line 55, in main
    train_loss = trainer.train_epoch(train_loader, epoch)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/georgeka/Desktop/uni/tooth project internship/xray-to-cbct/trainer.py", line 59, in train_epoch
    for i, (input, target) in enumerate(train_loader):
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1345, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/dataloader.py", line 1371, in _process_data
    data.reraise()
  File "/opt/homebrew/lib/python3.11/site-packages/torch/_utils.py", line 694, in reraise
    raise exception
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 308, in _worker_loop
    data = fetcher.fetch(index)
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 265, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 142, in collate
    return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 142, in <listcomp>
    return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 119, in collate
    return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/lib/python3.11/site-packages/torch/utils/data/_utils/collate.py", line 162, in collate_tensor_fn
    return torch.stack(batch, 0, out=out)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: stack expects each tensor to be equal size, but got [400, 400, 280] at entry 0 and [536, 536, 336] at entry 3
```


This error is thrown when I run the train.py with the corresponding parameters:

```
python3 train.py --data_root /Volumes/Georges\ NVME\ 2\ 1/data_xray_to_cbct --train_file /Volumes/Georges\ NVME\ 2\ 1/data_xray_to_cbct/train.csv --val_file /Volumes/Georges\ NVME\ 2\ 1/data_xray_to_cbct/val.csv --batch_size 8 --epochs 100 --learning_rate 1e-4 --input_size 128 --output_channel 1 --num_workers 4 --exp my_experiment --arch ReconNet --print_freq 10 --output_path ./output --resume best --loss l1 --optim adam --num_views 3 --init_gain 0.02 --init_type normal --weight_decay 0
```


# 7. Citation

```
@article{shen2019patient,
  title={Patient-specific reconstruction of volumetric computed tomography images from a single projection view via deep learning},
  author={Shen, Liyue and Zhao, Wei and Xing, Lei},
  journal={Nature biomedical engineering},
  volume={3},
  number={11},
  pages={880--888},
  year={2019},
  publisher={Nature Publishing Group}
}
```