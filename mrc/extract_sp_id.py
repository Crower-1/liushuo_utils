from mrc.io import get_tomo, save_tomo

def extract_sp_id(tomo_path, output_path, id):
    """Extract a specific ID from a tomogram and save it as a new MRC file.

    Args:
        tomo_path (str): Path to the input tomogram.
        output_path (str): Path to the output MRC file.
        id (int): The ID to extract.
    """
    # Load the tomogram
    tomo = get_tomo(tomo_path)

    # Extract the specific ID
    mask = (tomo == id)
    

    # Save the mask as an MRC file
    save_tomo(mask, output_path)

    print(f"ID {id} extracted and saved to {output_path}")
    
def extract_er(tomo_path):
    # Load the tomogram
    tomo = get_tomo(tomo_path)
    
    tomo_filename = tomo_path.split('/')[-1]
    er_path = tomo_path.replace(tomo_filename, 'Prediction/ER_nn.mrc')
    er_memb_path = tomo_path.replace(tomo_filename, 'Prediction/ER_memb.mrc')
    # Extract the specific ID
    # mask = (tomo == id)
    mask_er = (tomo == 1)
    # Save the mask as an MRC file
    save_tomo(mask_er, er_path)
    
    mask_er_memb = (tomo == 6)
    save_tomo(mask_er_memb, er_memb_path)
    
def extract_mito(tomo_path):
    # Load the tomogram
    tomo = get_tomo(tomo_path)
    
    tomo_filename = tomo_path.split('/')[-1]
    mito_path = tomo_path.replace(tomo_filename, 'Prediction/mito_nn.mrc')
    mito_memb_path = tomo_path.replace(tomo_filename, 'Prediction/mito_memb.mrc')
    # Extract the specific ID
    # mask = (tomo == id)
    mask_er = (tomo == 2)
    # Save the mask as an MRC file
    save_tomo(mask_er, mito_path)
    
    mask_er_memb = (tomo == 7)
    save_tomo(mask_er_memb, mito_memb_path)
    
def extract_MT(tomo_path):
    # Load the tomogram
    tomo = get_tomo(tomo_path)
    
    tomo_filename = tomo_path.split('/')[-1]
    MT_path = tomo_path.replace(tomo_filename, 'Prediction/MT.mrc')
    # mito_memb_path = tomo_path.replace(tomo_filename, 'Prediction/mito_memb.mrc')
    # Extract the specific ID
    # mask = (tomo == id)
    mask_MT = (tomo == 3) | (tomo == 8)
    # Save the mask as an MRC file
    save_tomo(mask_MT, MT_path)
    
def extract_memb(tomo_path):
    # Load the tomogram
    tomo = get_tomo(tomo_path)
    
    tomo_filename = tomo_path.split('/')[-1]
    memb_path = tomo_path.replace(tomo_filename, 'Prediction/memb.mrc')
    # mito_memb_path = tomo_path.replace(tomo_filename, 'Prediction/mito_memb.mrc')
    # Extract the specific ID
    # mask = (tomo == id)
    mask_memb = (tomo == 5)
    # Save the mask as an MRC file
    save_tomo(mask_memb, memb_path)
    
def extract_vesicle(tomo_path):
    # Load the tomogram
    tomo = get_tomo(tomo_path)
    
    tomo_filename = tomo_path.split('/')[-1]
    vesicle_path = tomo_path.replace(tomo_filename, 'Prediction/vesicle.mrc')
    # mito_memb_path = tomo_path.replace(tomo_filename, 'Prediction/mito_memb.mrc')
    # Extract the specific ID
    # mask = (tomo == id)
    mask_vesicle = (tomo == 9)
    # mask_vesicle = (tomo == 4) | (tomo == 9)
    # Save the mask as an MRC file
    save_tomo(mask_vesicle, vesicle_path)
    
def extract_actin(tomo_path):
    # Load the tomogram
    tomo = get_tomo(tomo_path)
    
    tomo_filename = tomo_path.split('/')[-1]
    actin_path = tomo_path.replace(tomo_filename, 'Prediction/actin.mrc')
    # mito_memb_path = tomo_path.replace(tomo_filename, 'Prediction/mito_memb.mrc')
    # Extract the specific ID
    # mask = (tomo == id)
    mask_actin = (tomo == 10)
    # Save the mask as an MRC file
    save_tomo(mask_actin, actin_path)

    
tomo_path = f"/media/liushuo/data1/data/synapse_seg/pp463/ret1_10tomo.mrc"
# extract_er(tomo_path)
# extract_mito(tomo_path)
# extract_MT(tomo_path)
# extract_memb(tomo_path)
# extract_vesicle(tomo_path)
extract_actin(tomo_path)

# tomo_path = f"/media/liushuo/data1/data/fig_demo_2/pp3972/ret1.mrc"
# extract_sp_id(tomo_path, f"/media/liushuo/data1/data/fig_demo_2/pp199/synapse_seg/pp199/er/pp199_er_memb_label.mrc", 150)
    
# tomo_path = f"/media/liushuo/data1/data/fig_demo_2/p184/ret1_crop.mrc"
# memb_output_path = f"/media/liushuo/data1/data/fig_demo_2/p184/mt/mt.mrc"
# extract_sp_id(tomo_path, memb_output_path, 7)
# er_output_path = f"/media/liushuo/data1/data/fig_demo/pp518/er/pp518_er_label.mrc"
# empty_output_path = f"/media/liushuo/data1/data/tcl_demo/pp0039/ER/pp0039_ER_label.mrc"

# def generate_paths(base_name):
#     tomo_path = f"/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/{base_name}/ret1.mrc"
#     output_path = f"/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/{base_name}/membrane/{base_name}_membrane_label2.mrc"
#     empty_output_path = f"/media/liushuo/data1/data/actin trace/actintomo/20200820/tomo/{base_name}/ER/{base_name}_ER_label.mrc"
    
#     return tomo_path, output_path, empty_output_path

# # 示例调用
# base_name = "p255"

# working_dir = '/media/liushuo/data1/data/fig_demo_2/'
# base_names = ['p90']
# class_name = 'membrane'
# for base_name in base_names:
#     tomo_path = f"{working_dir}/{base_name}/ret1.mrc"
#     memb_output_path = f"{working_dir}/{base_name}/{class_name}/{base_name}_{class_name}_label.mrc"
#     extract_sp_id(tomo_path, memb_output_path, 5)
# tomo_path = f"/media/liushuo/data1/data/fig_demo_2/{base_name}/ret1.mrc"
# memb_output_path = f"/media/liushuo/data1/data/fig_demo_2/{base_name}/{class_name}/{base_name}_{class_name}_label.mrc"
# tomo_path, output_path, empty_output_path = generate_paths(base_name)

# tomo_path = '/media/liushuo/data1/data/fig_demo_2/p193/mito/p193_mito_nn.mrc'
# memb_output_path = '/media/liushuo/data1/data/fig_demo_2/p193/mito/p193_mito_1.mrc'
# extract_sp_id(tomo_path, memb_output_path, 1)
# extract_sp_id(tomo_path, er_output_path, 3)
# extract_sp_id(tomo_path, empty_output_path, 7)