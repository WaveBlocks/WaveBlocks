#!/bin/bash
source_dir=$1;
dest_dir=$2;

omit_extension="hdf5";

pushd $source_dir;

for file in $(find | grep -v "\.$omit_extension");
do
    if [ -f "$file" ]; then
        mkdir -p $dest_dir/$(dirname $file);
        cp $file $dest_dir/$(dirname $file);
    fi;
done

popd;
