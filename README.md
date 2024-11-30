# Fluid simulation
My CPU implementation of fluid based on what is described in [GPU Gems](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu).


## Renderings

Normal                     |Multiply velocity by scalar after advection           
:-------------------------:|:-------------------------:
![](https://cabbache.github.io/fluid3.gif)  |  ![](https://cabbache.github.io/fluid4.gif)

## How to run on linux

* Make sure libsdl2 installed. For debian based OS: `sudo apt update && sudo apt install libsdl2-dev -y`
* Make sure g++, cmake and make are installed
* `mkdir build && cd build && cmake .. && make && ./fluid`

## Notes
* Still need to implement boundary conditions

## Reference

- https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
- https://wiki.libsdl.org/SDL2/FrontPage
