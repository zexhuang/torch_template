#!/bin/bash

cat <<EOF > gdl4spatial.yaml
name: geoai
channels:
  - conda-forge
  - defaults
dependencies:
  # --- Core Python ---
  - python=3.11
  - cmake=3.27

  # --- Scientific & Data ---           
  - scikit-learn
  - numba                      
  - scipy
  - pandas
  - jupyterlab
  - ipykernel
  - ipywidgets
  - pyyaml
  - pycocotools
  - tqdm    

  # --- Geospatial Core ---
  - gdal
  - pdal
  - python-pdal
  - geopandas
  - shapely           
  - fiona
  - pyproj
  - rtree
  - h3-py
  - pysal
  - esda
  - momepy
  - spopt
  - osmnx
  - rasterio
  - rioxarray
  - rio-tiler
  - xarray-spatial
  - geocube
  - stackstac

  # --- 3D Processing & Visualization ---
  - trimesh
  - pymeshlab
  - pyvista
  - trame
  - plyfile
  - open3d
  - matplotlib
  - seaborn
  - folium        
EOF

echo "Building the Conda foundation..."
mamba env create -f gdl4spatial.yaml

echo "Installing Pip packages"

# PyTorch Foundation (CUDA 12.8 Build)
echo "Installing PyTorch Foundation..."
mamba run -n geoai pip install torch==2.10.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# HuggingFace & ML Ops Tools
echo "Installing ML Ops & HuggingFace Core..."
mamba run -n geoai pip install lightning torchinfo torchmetrics accelerate einops wandb tensorboard hydra-core torchgeo kornia datasets transformers tokenizers

# TDA & Spatial Geometry
echo "Installing TDA & Spatial Libraries..."
mamba run -n geoai pip install city2graph TorchSpatial giotto-tda topomodelx torch_topological tadasets toponetx gudhi persim laspy lazrs startinpy fast-simplification 

# PyG Extensions
echo "Installing PyG C++ Extensions..."
mamba run -n geoai pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
