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

## tricks make your generation better
### fusion method leverages many id images to enhance fidelity
1. Besides mean fusion, you can try max or max_token, which can boost some major feature of a face (like large eyes, special nose or sth). it can go distortion beyond fidelity though.
2. With train_weight method, you can train with less than 2000 steps to make a deeper fusion than the non-training methods. Be aware too many training steps will make the training crash to the prior image.

### additional notes
1. Flux is a high capacity base model, it even can cognize the input image in some super human way. 
for example, you can resize your high quality input image with lanczos method rather than nearest area or billinear. you get finer texture. Keep in mind that taking care of your input image is the thing when the base model is strong.
2. The best pulid weight is around 0.8-0.95 for flux pulid 0.9.0. 1.0 is not good. For 0.9.1, it's higher towards around 0.9-1.0. Nonetheless the 0.9.1 is not always better than 0.9.0.
3. The base model is flux-dev or its finetuning, and the precision does mean the thing. fp16 should always be sound. fp8 is OK. I won't recommend gguf or nf4 things.
4. Some of the finetuned flux dev model may have strong bias. for example, it may sway the faces to a certain human race.
5. Euler simple is always working. Euler beta give you higher quality especially if your input image is somewhat low quality.
6. If you wanna use 3rd party flux-d weight, better to use a merged one or with a lora weight, rather than a finetuned one. Full finetuning can hurt the connection between pulid and original flux-d base model. You can test by yourself though. 

## basic notes for common users
This is an experimental node. It can give enhanced result but I'm not promising basic instructions for users who barely know about python developing or AI developing.

Please follow the comfyui instructions or https://github.com/balazik/ComfyUI-PuLID-Flux to enable usage.

If you are just using SDXL pulid, you can use https://github.com/cubiq/PuLID_ComfyUI. Some of the installation instructions there may also help.
