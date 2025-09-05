To use:
conda activate ssl-cifar

# update dependency
conda env export --from-history > environment.yml
# add new package
conda install -n ssl-cifar new_package 