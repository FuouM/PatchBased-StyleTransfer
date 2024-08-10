# PatchBased-StyleTransfer
Few-Shot Patch-Based Training, easy.

Interactive Video Stylization Using Few-Shot Patch-Based Training - https://github.com/OndrejTexler/Few-Shot-Patch-Based-Training

Few Shot Patch Based Training for Image Translation using PyTorch Lightning - https://github.com/rnwzd/FSPBT-Image-Translation

WORK IN PROGRESS!!!

## Work pipeline

1. Get style keyframes
2. Get style mask frames (TODO: make gauss mixture works with entire image)
3. Generate grouth-truth using EbSynth (for all frames)
4. Get flows from EbSynth (FWD and BWD)
5. Train a model 
6. Use model to generate new

From observations it seems like patch-based is better for foreground objects, while EbSynth can do background. So if you want to style transfer the whole thing, use EbSynth GT as background and patch-based output as foreground.

## Setup

`python.exe -m pip install .`

Included is a Python wrapper for poisson_disk_sampling: `python.exe poisson_setup.py build_ext --inplace`


```
ffmpeg -framerate 24 -i image_%05d.png -c:v libx264 -crf 5 -pix_fmt yuv420p output.mp4
```

## Credits

https://github.com/thinks/poisson-disk-sampling

```
@Article{Texler20-SIG,
    author    = "Ond\v{r}ej Texler and David Futschik and Michal Ku\v{c}era and Ond\v{r}ej Jamri\v{s}ka and \v{S}\'{a}rka Sochorov\'{a} and Menglei Chai and Sergey Tulyakov and Daniel S\'{y}kora",
    title     = "Interactive Video Stylization Using Few-Shot Patch-Based Training",
    journal   = "ACM Transactions on Graphics",
    volume    = "39",
    number    = "4",
    pages     = "73",
    year      = "2020",
}
```