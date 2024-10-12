# ComfyUI-PuLID-Flux-Enhanced
adapted from https://github.com/balazik/ComfyUI-PuLID-Flux

workflow: see example flux_pulid_multi.json

## new features
### common fusion methods for multi-image input
mean(official), concat, max...etc

### some further experimental fusion methods.
using the norm of the conditions to weight them

using the max norm token among images

a novel very fast embeddings self-training methods...etc

### switch between using gray image (official) and rgb.
in some cases, using gray image will bring detail loss
