# re-blocking

# (1) Image Generation

## Instructions

1. Clone our [re-blocking Github repo](https://github.com/DrawingTogether/re-blocking)
2. create an *.env* file in the main folder (hidden file?)
    1. create and enter [Mapbox Access Token](https://docs.mapbox.com/api/overview/) for the Mapbox Web API
    2. enter your file path to the parcel and building data ([NYC is on our Dropbox](https://www.dropbox.com/sh/zuemdhfftqv0ksv/AABh6hbieVlcccw-h94zu939a?dl=0))
    
    ```
    MAPBOX_ACCESS_TOKEN="$YOUR-API-KEY"
    LOCAL_PATH="$YOUR-DROPBOX-PATH/Million Neighborhoods/"
    ```
    
3. Run the Notebook *parcels-buildings.ipynb to e*xport images (parcels and buildings)
4. Combine the images with the script *combine_images.sh* for Pix2Pix HD (leave the individual ones for CycleGan)

---

# (2) Model Training

## Instructions

1. Clone the [model’s Github repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
    1. Read the *readme.md*
    2. install all dependencies (see readme)
    3. install/update *cuda* for linux
2. Move the generated images (from step 1 or from Dropbox → [Images](https://www.dropbox.com/sh/v3258q791pu83qg/AAA4rnleCXfD8qcerYGwqc0_a?dl=0)) into */datasets/parcels*
    1. *→/test, /train, and /val for Pix2Pix*
    2. TestA, TestB, TrainA, TrainB etc. for CycleGan
3. Open the repo folder in your terminal
    
    ```bash
    cd …/pytorch-CycleGAN-and-pix2pix/
    ```
    
4. Train the model
    1. (I named the output folder according to the parameters I used not to forget them…)
    
    ```bash
    **Pix2Pix HD**
    python3 [train.py](http://train.py/) --dataroot ./datasets/parcels --name test --model pix2pix --direction AtoB --n_epochs 100 --batch_size 60
    
    **CycleGan**
    python3  --dataroot ./datasets/parcels --name parcels_cycle-gan_25k_e50_b50_A-B --model cycle_gan --direction AtoB --n_epochs 10 --batch_size 1
    ```
    
5. Output
    1. See output in */checkpoints/FOLDERNAME/web/index.html*
    2. Monitor the training logs with *logs-visualised.ipynb* (WIP)
    3. Copy your folder (without the training data) to our Dropbox → [Model-Training](https://www.dropbox.com/sh/c9iiuyuvuga7gdj/AABML9R5Rk3U9Z1ylRtDLi-0a?dl=0) if you want to share the results 😊

### Parameters

- See */options → base_options.py, test_options.py, and train_options.py* for the full parameter list (and to set standards)
- So far, the goal has been to increase *batch_size to* just below memory collapse

## Other

```bash
watch -n0.1 nvidia-smi → Nvidia GPU monitor
```

---

# (3) Running the trained Model

## Instructions

- Find the trained model here
    - CycleGan → [Dropbox](https://www.dropbox.com/sh/4qdrhgwmew3yy6u/AAB7zd3onxzV1RtHgAGREsSga?dl=0)
        - Trained on 25k Images (Train A/B: 20k, Buffer: 2.5k, Val: 2.5k)
        - 50 Epochs, batch Size 50
        - Direction A→B
    - Pix2Pix → needs to be retrained due to storage issue

---

# Notes

## Running (Training?) the model on an Apple ARM (M1/M2) Mac

### **Problem**

- Issue: No NVIDIA GPU → NO CUDA
- ARM GPU (MPS) can generally be used with PyTorch but doesn’t seem to work with our model

### Solution

- Run on CPU → add to terminal command when running training script: “--gpu_ids -1”

```bash
python3 [train.py](http://train.py/) --dataroot ./datasets/parcels --name facades_pix2pix --model pix2pix --direction BtoA **--gpu_ids -1**
```

### Resources

- https://developer.apple.com/metal/pytorch/
- [https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/?ref=mrdbourke.com)
- https://www.mrdbourke.com/pytorch-apple-silicon/