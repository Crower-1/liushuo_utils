import mrcfile
import numpy as np
import json
import os
import xml.etree.ElementTree as ET

def get_tomo(path):
    """
    Load a 3D MRC file as a numpy array.
    """
    with mrcfile.open(path) as mrc:
        data = mrc.data
    return data

def save_tomo(data, path, voxel_size=17.14):
    """
    Save a 3D numpy array as an MRC file.
    """
    with mrcfile.new(path, overwrite=True) as mrc:
        data = data.astype(np.int8)
        mrc.set_data(data)
        mrc.voxel_size = voxel_size

def load_json(json_path):
    """
    Load vesicle information from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['vesicles']

def get_vesicle_id_from_center(tomo_data, center):
    """
    Get the vesicle ID from tomo_data using the center coordinates.

    Parameters:
    - tomo_data: ndarray
        The 3D segmentation mask data.
    - center: list or tuple of three floats
        The (z, y, x) coordinates of the vesicle center.

    Returns:
    - vesicle_id: int
        The ID of the vesicle at the given center coordinates.
    """
    z, y, x = center
    # Round the coordinates to the nearest integer
    z = int(round(z))
    y = int(round(y))
    x = int(round(x))
    
    # Ensure the coordinates are within the bounds of tomo_data
    if (0 <= z < tomo_data.shape[0] and
        0 <= y < tomo_data.shape[1] and
        0 <= x < tomo_data.shape[2]):
        vesicle_id = tomo_data[z, y, x]
        return vesicle_id
    else:
        raise ValueError(f"Center coordinates {center} are out of bounds for tomo_data with shape {tomo_data.shape}")

def categorize_vesicles(vesicles, tomo_data, threshold=34, contact_id_list=None, tether_id_list=None):
    """
    Categorize vesicles into small and other based on average radius,
    excluding contact and tether vesicles.

    Parameters:
    - vesicles: list of dict
        The list of vesicle information.
    - tomo_data: ndarray
        The 3D segmentation mask data.
    - threshold: float
        The radius threshold to categorize vesicles.
    - contact_id_list: list of int, optional
        List of contact vesicle IDs to exclude.
    - tether_id_list: list of int, optional
        List of tether vesicle IDs to exclude.

    Returns:
    - small_vesicle_ids: list of int
        List of IDs for small vesicles.
    - other_vesicle_ids: list of int
        List of IDs for other vesicles.
    """
    small_vesicle_ids = []
    other_vesicle_ids = []
    
    for vesicle in vesicles:
        radii = vesicle['radii']
        avg_radius = np.mean(radii)
        
        center = vesicle['center']
        try:
            vesicle_id = get_vesicle_id_from_center(tomo_data, center)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        
        # Exclude contact and tether vesicles from small and other
        if contact_id_list and vesicle_id in contact_id_list:
            continue
        if tether_id_list and vesicle_id in tether_id_list:
            continue
        
        if avg_radius < threshold:
            small_vesicle_ids.append(vesicle_id)
        else:
            other_vesicle_ids.append(vesicle_id)
    
    return small_vesicle_ids, other_vesicle_ids

def create_mask(tomo_data, vesicle_ids):
    """
    Create a mask for the given vesicle IDs.

    Parameters:
    - tomo_data: ndarray
        The 3D segmentation mask data.
    - vesicle_ids: list of int
        The list of vesicle IDs to include in the mask.

    Returns:
    - mask: ndarray
        A binary mask where voxels belonging to the specified vesicle IDs are 1, else 0.
    """
    mask = np.isin(tomo_data, vesicle_ids).astype(np.float32)
    return mask

def generate_vesicle_mrc(mask, output_path):
    """
    Save the vesicle mask as an MRC file.
    """
    save_tomo(mask, output_path)

def process_vesicle_data(mrc_path, json_path, small_vesicle_output, other_vesicle_output, contact_id_output=None, tether_id_output=None, threshold=34, contact_id_list=None, tether_id_list=None):
    """
    Process vesicle data and save small, other, contact, and tether vesicle MRC files.

    Parameters:
    - mrc_path: str
        Path to the input MRC segmentation mask.
    - json_path: str
        Path to the JSON file containing vesicle information.
    - small_vesicle_output: str
        Path to save the small vesicles MRC file.
    - other_vesicle_output: str
        Path to save the other vesicles MRC file.
    - contact_id_output: str, optional
        Path to save vesicles with specific contact IDs.
    - tether_id_output: str, optional
        Path to save vesicles with specific tether IDs.
    - threshold: float
        Radius threshold to categorize vesicles.
    - contact_id_list: list of int, optional
        List of contact vesicle IDs to extract and save in contact_id_output.
    - tether_id_list: list of int, optional
        List of tether vesicle IDs to extract and save in tether_id_output.
    """
    
    # Ensure output directories exist
    small_vesicle_dir = os.path.dirname(small_vesicle_output)
    other_vesicle_dir = os.path.dirname(other_vesicle_output)
    
    if not os.path.exists(small_vesicle_dir):
        os.makedirs(small_vesicle_dir)
        print(f"Created directory: {small_vesicle_dir}")
    
    if not os.path.exists(other_vesicle_dir):
        os.makedirs(other_vesicle_dir)
        print(f"Created directory: {other_vesicle_dir}")
    
    if contact_id_output and not os.path.exists(os.path.dirname(contact_id_output)):
        os.makedirs(os.path.dirname(contact_id_output))
        print(f"Created directory: {os.path.dirname(contact_id_output)}")
        
    if tether_id_output and not os.path.exists(os.path.dirname(tether_id_output)):
        os.makedirs(os.path.dirname(tether_id_output))
        print(f"Created directory: {os.path.dirname(tether_id_output)}")

    # Load the MRC file
    print("Loading MRC file...")
    tomo_data = get_tomo(mrc_path)
    print("MRC file loaded.")

    # Load the vesicle information from JSON
    print("Loading JSON file...")
    vesicles = load_json(json_path)
    print("JSON file loaded.")

    # Categorize vesicles based on average radius, excluding contact and tether IDs
    print("Categorizing vesicles...")
    small_vesicle_ids, other_vesicle_ids = categorize_vesicles(vesicles, tomo_data, threshold, contact_id_list, tether_id_list)
    print(f"Found {len(small_vesicle_ids)} small vesicles and {len(other_vesicle_ids)} other vesicles.")

    # Create masks
    print("Creating masks for small vesicles...")
    small_vesicles_mask = create_mask(tomo_data, small_vesicle_ids)
    print("Creating masks for other vesicles...")
    other_vesicles_mask = create_mask(tomo_data, other_vesicle_ids)

    # Save the masks as MRC files
    print(f"Saving small vesicles to {small_vesicle_output}...")
    generate_vesicle_mrc(small_vesicles_mask, small_vesicle_output)
    print(f"Saving other vesicles to {other_vesicle_output}...")
    generate_vesicle_mrc(other_vesicles_mask, other_vesicle_output)

    # If contact IDs are provided, save those vesicles as well
    if contact_id_output and contact_id_list:
        print(f"Saving specific contact vesicles with IDs {contact_id_list} to {contact_id_output}...")
        contact_vesicle_mask = create_mask(tomo_data, contact_id_list)
        generate_vesicle_mrc(contact_vesicle_mask, contact_id_output)

    # If tether IDs are provided, save those vesicles as well
    if tether_id_output and tether_id_list:
        print(f"Saving specific tether vesicles with IDs {tether_id_list} to {tether_id_output}...")
        tether_vesicle_mask = create_mask(tomo_data, tether_id_list)
        generate_vesicle_mrc(tether_vesicle_mask, tether_id_output)

    print("Processing complete.")

def extract_others_vesicle_ids(xml_path):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 存储'others'类型的vesicleId
    others_vesicle_ids = []
    
    # 遍历所有Vesicle标签
    for vesicle in root.findall('Vesicle'):
        vesicle_id = vesicle.get('vesicleId')  # 获取vesicleId
        
        # 查找Type标签
        type_tag = vesicle.find('Type')
        if type_tag is not None and type_tag.get('t') == 'others':
            others_vesicle_ids.append(int(vesicle_id))  # 添加vesicleId到列表
    
    # 返回'others'类型的vesicleId列表
    return others_vesicle_ids

def generate_vesicle_paths(base_name):
    mrc_path = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/{base_name}_label_vesicle.mrc'
    json_path = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/{base_name}_vesicle.json'
    xml_path = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/vesicle_analysis/{base_name}_filter.xml'
    small_vesicle_output = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/filter_vesicle/small_vesicles.mrc'
    other_vesicle_output = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/filter_vesicle/other_vesicles.mrc'
    contact_id_output = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/filter_vesicle/contact_vesicles.mrc'
    tether_id_output = f'/media/liushuo/data1/data/fig_demo/{base_name}/ves_seg/filter_vesicle/tether_vesicles.mrc'
    
    return mrc_path, json_path, xml_path, small_vesicle_output, other_vesicle_output, contact_id_output, tether_id_output



if __name__ == "__main__":
    # Example usage
    # mrc_path = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/pp0039_label_vesicle.mrc'
    # json_path = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/pp0039_vesicle.json'
    # xml_path = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/vesicle_analysis/pp0039_filter.xml'
    # small_vesicle_output = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/filter_vesicle/small_vesicles.mrc'
    # other_vesicle_output = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/filter_vesicle/other_vesicles.mrc'
    # contact_id_output = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/filter_vesicle/contact_vesicles.mrc'
    # tether_id_output = '/media/liushuo/data1/data/tcl_demo/pp0039/ves_seg/filter_vesicle/tether_vesicles.mrc'

    base_name = "pp1776"
    mrc_path, json_path, xml_path, small_vesicle_output, other_vesicle_output, contact_id_output, tether_id_output = generate_vesicle_paths(base_name)



    threshold = 34 / 1.714 / 2
    
    init_tether_id_list = extract_others_vesicle_ids(xml_path)
    contact_id_list = []  # Example contact vesicle IDs
    tether_id_list = [id for id in init_tether_id_list if id not in contact_id_list]
    # tether_id_list = [2, 4, 6]  # Example tether vesicle IDs
    process_vesicle_data(mrc_path, json_path, small_vesicle_output, other_vesicle_output, contact_id_output, tether_id_output, threshold, contact_id_list, tether_id_list)
