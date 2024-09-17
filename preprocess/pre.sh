for zip_file in datasets/dataset_dara/clinic_files/*.zip; do
    unzip -o "$zip_file" -d /tmp/unzipped_files
    cp -r /tmp/unzipped_files/* datasets/dataset_dara/clinic_temp/
    rm -rf /tmp/unzipped_files/*
done

