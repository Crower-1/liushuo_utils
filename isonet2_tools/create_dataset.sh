#!/usr/bin/env bash
set -euo pipefail
#
# create_dataset.sh
# Build an IsoNet2 dataset from a stack-out directory.
# Steps:
# 1) Link .ali/.tlt/.xtilt/tilt.com into RAW_ALI_DATA.
# 2) Split each tilt series into ODD/EVN subsets.
# 3) Reconstruct ODD/EVN with IMOD and post-process outputs.
# 4) Link final volumes into dataset-level ODD/EVN folders.
#
# Usage:
#   ./create_dataset.sh [stack_out_dir] [base_names_input] [isonet_dataset_dir]
#   base_names_input can be comma- or space-separated.

module purge
module load imod/4.12.16

# 0. Create RAW_ALI_DATA symlinks, generate odd/even splits, and reconstruct.

stack_out_dir="/share/data/CryoET_Data/synapse/synapse202008/20200820_20200731_g2b2_65_trig/stack-out-pp"
base_names_input="pp0312"
isonet_dataset_dir="/share/data/CryoET_Data/liushuo/dataset/IsoNet2/pp0312_bin8_2"

# 0.1. Read inputs if not provided.
if [[ -z "${stack_out_dir}" ]]; then
  read -r -p "stack-out-dir: " stack_out_dir
fi
if [[ -z "${base_names_input}" ]]; then
  read -r -p "base_names (comma or space separated): " base_names_input
fi
if [[ -z "${isonet_dataset_dir}" ]]; then
  read -r -p "isonet_dataset_dir: " isonet_dataset_dir
fi

# 1. Create RAW_ALI_DATA and link expected files.
raw_ali_dir="${isonet_dataset_dir}/RAW_ALI_DATA"
mkdir -p "${raw_ali_dir}"

base_names_input="${base_names_input//,/ }"
for base_name in ${base_names_input}; do
  src_dir="${stack_out_dir}/${base_name}"
  ln -sfn "${src_dir}/${base_name}.ali" "${raw_ali_dir}/${base_name}.ali"
  ln -sfn "${src_dir}/${base_name}.tlt" "${raw_ali_dir}/${base_name}.tlt"
  ln -sfn "${src_dir}/${base_name}.xtilt" "${raw_ali_dir}/${base_name}.xtilt"
  ln -sfn "${src_dir}/tilt.com" "${raw_ali_dir}/${base_name}_tilt.com"
  echo "linked ${base_name} into ${raw_ali_dir}"
done

# 2. Generate odd/even folder.
cd "${isonet_dataset_dir}"
python generate_even_odd.py ./

# 3. Reconstruct ODD/EVEN from tilt.com.
# image_bin should match IMAGEBINNED in tilt.com.
image_bin=4
base_names_input="${base_names_input//,/ }"
for base_name in ${base_names_input}; do
  for subset in ODD EVEN; do
    subset_dir="${raw_ali_dir}/${subset}"
    if [[ ! -d "${subset_dir}" ]]; then
      echo "Skipping ${base_name} ${subset}: ${subset_dir} not found"
      continue
    fi

    pushd "${subset_dir}" >/dev/null
    tilt_com="${base_name}_tilt.com"
    tilt_com2="${base_name}_tilt2.com"
    if [[ ! -f "${tilt_com}" ]]; then
      echo "Skipping ${base_name} ${subset}: missing ${tilt_com}"
      popd >/dev/null
      continue
    fi

    sed '/\$tilt/d;/\$if/d' "${tilt_com}" > "${tilt_com2}"

    if [[ "${subset}" == "ODD" ]]; then
      suffix="ODD"
    else
      suffix="EVN"
    fi

    sed -i \
      -e "s/^InputProjections .*/InputProjections ${base_name}_${suffix}.mrc/" \
      -e "s/^OutputFile .*/OutputFile ${base_name}_${suffix}_full.rec/" \
      -e "s/^TILTFILE .*/TILTFILE ${base_name}_${suffix}.tlt/" \
      -e "s/^XTILTFILE .*/XTILTFILE ${base_name}_${suffix}.xtilt/" \
      "${tilt_com2}"

    tilt -param "${tilt_com2}"

    wbpRec="${base_name}_${suffix}_full.rec"
    mrcbyte "${wbpRec}" "${wbpRec}"
    trimvol -rx "${wbpRec}" "${base_name}_bin${image_bin}_wbp_${suffix}.mrc" >>/dev/null
    rm "${wbpRec}"
    rm -f *~
    popd >/dev/null
  done
done

# 4. Link reconstructed volumes to dataset-level ODD/EVN folders.
odd_link_dir="${isonet_dataset_dir}/ODD"
evn_link_dir="${isonet_dataset_dir}/EVN"
mkdir -p "${odd_link_dir}" "${evn_link_dir}"
base_names_input="${base_names_input//,/ }"
for base_name in ${base_names_input}; do
  odd_src="${raw_ali_dir}/ODD/${base_name}_bin${image_bin}_wbp_ODD.mrc"
  evn_src="${raw_ali_dir}/EVEN/${base_name}_bin${image_bin}_wbp_EVN.mrc"
  if [[ -f "${odd_src}" ]]; then
    ln -sfn "${odd_src}" "${odd_link_dir}/$(basename "${odd_src}")"
  else
    echo "Missing ODD output for ${base_name}: ${odd_src}"
  fi
  if [[ -f "${evn_src}" ]]; then
    ln -sfn "${evn_src}" "${evn_link_dir}/$(basename "${evn_src}")"
  else
    echo "Missing EVN output for ${base_name}: ${evn_src}"
  fi
done
