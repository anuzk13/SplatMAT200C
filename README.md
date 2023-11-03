# WebGL Gaussian Splat Viewer

_Spectacular AI fork_. See original README at https://github.com/antimatter15/splat/blob/main/README.md

## Usage

 1. Clone this repository and cd to its root folder
 2. `mkdir data`
 3. Extract some `.splat` files to `data/` (e.g., `example.splat`)
 4. Start a local webserver `python3 -m http.server 8000 --bind 127.0.0.1`
 5. Open in the browser: http://127.0.0.1:8000/?url=http://127.0.0.1:8000/data/example.splat&z_is_up=true \
    (or http://127.0.0.1:8000/?url=http://127.0.0.1:8000/data/example.splat if the model is oriented in a Y-is-up coordinate system)

## Controls

THREE JS mouse orbit controls, turntable rotation, assuming Y-is-up by default

## Splat files

Like in the original code, these are Gaussian Splats with flattened spherical harmonic (zeroth order coefficient only).
These can be converted from, e.g., Parquet files using these scripts: https://github.com/SpectacularAI/point-cloud-tools
