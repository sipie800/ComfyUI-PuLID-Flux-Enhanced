# ComfyUI-PuLID-Flux-Enhanced
adapted from https://github.com/balazik/ComfyUI-PuLID-Flux

workflow: see example flux_pulid_multi.json

## new features
### common fusion methods for multi-image input
mean(official), concat, max...etc

### some further experimental fusion methods.
using the norm of the conditions to weight them

using the max norm token among images

a novel very fast embeddings self-training methods(explained here: https://github.com/balazik/ComfyUI-PuLID-Flux/issues/28)...etc

### switch between using gray image (official) and rgb.
in some cases, using gray image will bring detail loss

![2024-10-12_204047](https://github.com/user-attachments/assets/0ae96170-2eff-44e9-a53a-6a7447dbc0f1)


