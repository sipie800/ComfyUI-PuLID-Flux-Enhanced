# ComfyUI-PuLID-Flux-Enhanced
adapted from https://github.com/balazik/ComfyUI-PuLID-Flux

workflow: see example flux_pulid_multi.json

## update oct.28 2024
Add an optional prior image input for the node. When using the train_weight method, the prior image will act as the main id image, which will lead the other id images to sum up to an optimized id embedding.

This prior was randomly choosen previously, now we can assign it.

Leaving the prior image input empty is OK just as previous.

Please choose the best id image in your mind as the prior, or just experiment around and see what happens.
![oct28](https://github.com/user-attachments/assets/6a481cd9-2836-4f6f-9ad5-7458356c332a)

## new features
### common fusion methods for multi-image input
mean(official), concat, max...etc

### some further experimental fusion methods.
using the norm of the conditions to weight them

using the max norm token among images

a novel very fast embeddings self-training methods(explained here: https://github.com/balazik/ComfyUI-PuLID-Flux/issues/28)

### switch between using gray image (official) and rgb.
in some cases, using gray image will bring detail loss

![2024-10-12_204047](https://github.com/user-attachments/assets/0ae96170-2eff-44e9-a53a-6a7447dbc0f1)


