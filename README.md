# Fluid simulation
My CPU implementation of fluid based on what is described in [GPU Gems](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu).


## Renderings

Normal                     |Multiply velocity by scalar after advection           
:-------------------------:|:-------------------------:
![](https://cabbache.github.io/fluid3.gif)  |  ![](https://cabbache.github.io/fluid4.gif)


## Note
- Still need to implement boundary conditions
- It requires libsdl2

## Reference

- https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
- https://wiki.libsdl.org/SDL2/FrontPage
