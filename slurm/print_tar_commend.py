base_name = 'liushuo'
cmd = (
    f"tar -cvf ./data_transfer/{base_name}.tar ./{base_name} "
    f"&& chown :cryoetimaging ./data_transfer/{base_name}.tar "
    f"&& chmod 775 ./data_transfer/{base_name}.tar"
)
print(cmd)