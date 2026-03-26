- Create Dataset
    ```bash
    !python create_dataset.py --size 100000
    ```

- Train model from scratch
    ```bash
    !python train.py --epochs 5
    ```

- Resume training
    ```bash
    !python train.py --epochs 5 --resume
    ```