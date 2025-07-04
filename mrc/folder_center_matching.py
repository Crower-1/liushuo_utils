import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def parse_arguments():
    parser = argparse.ArgumentParser(description="Aggregate center matching results across multiple tomograms.")
    parser.add_argument("current_path", type=str, help="Path to the current directory containing tomo directories.")
    parser.add_argument("--broken_json_path", type=str, default=None, help="Path to segVesicle_heart_broken.json file (optional).")
    parser.add_argument("output_xml", type=str, help="Path to the output XML file.")
    return parser.parse_args()

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def load_broken_json(broken_json_path):
    """
    Load the broken JSON file if it exists.
    Returns a dictionary mapping tomo_name to a boolean indicating if it's broken.
    If the file doesn't exist, returns an empty dictionary.
    """
    if broken_json_path is None:
        print("No broken JSON path provided. Proceeding without skipping any tomograms.")
        return {}

    if not os.path.isfile(broken_json_path):
        print(f"Warning: Broken JSON file '{broken_json_path}' does not exist. Proceeding without skipping any tomograms.")
        return {}
    try:
        with open(broken_json_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Warning: Broken JSON file '{broken_json_path}' is improperly formatted. Details: {e}. Proceeding without skipping any tomograms.")
        return {}
    except Exception as e:
        print(f"Warning: Unable to read broken JSON file '{broken_json_path}'. Details: {e}. Proceeding without skipping any tomograms.")
        return {}

def extract_base_tomo_name(tomo_name):
    return tomo_name.split('-1')[0] if '-1' in tomo_name else tomo_name

def run_center_matching(label_path, json_path):
    python_executable = "/share/home/liushuo/.conda/envs/npr/bin/python"
    script_path = "/share/data/CryoET_Data/liushuo/utils/center_matching.py"
    cmd = [python_executable, script_path, label_path, json_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: center_matching.py failed for '{label_path}' and '{json_path}'. Details: {e.stderr}")
        return None

def parse_center_matching_output(output):
    """
    Parses the output from center_matching.py to extract required statistics.
    Assumes the output contains specific lines with identifiable patterns.
    """
    if output is None:
        return None

    stats = {}
    lines = output.splitlines()
    for line in lines:
        if "Number of unmatched mask IDs:" in line:
            try:
                stats['unmatched_masks_count'] = int(line.split(":")[1].strip())
            except ValueError:
                stats['unmatched_masks_count'] = 0
        elif "Percentage of unmatched mask IDs:" in line:
            try:
                stats['unmatched_masks_percent'] = float(line.split(":")[1].strip().replace('%', ''))
            except ValueError:
                stats['unmatched_masks_percent'] = 0.0
        elif "List of unmatched mask IDs:" in line:
            ids_str = line.split(":")[1].strip().strip('[]')
            # Convert to integers, even if the IDs are float-like
            stats['unmatched_mask_ids'] = [int(float(id_.strip())) for id_ in ids_str.split(',')] if ids_str else []
        elif "Number of unmatched JSON vesicle IDs:" in line:
            try:
                stats['unmatched_vesicles_count'] = int(line.split(":")[1].strip())
            except ValueError:
                stats['unmatched_vesicles_count'] = 0
        elif "Percentage of unmatched JSON vesicle IDs:" in line:
            try:
                stats['unmatched_vesicles_percent'] = float(line.split(":")[1].strip().replace('%', ''))
            except ValueError:
                stats['unmatched_vesicles_percent'] = 0.0
        elif "List of unmatched JSON vesicle IDs:" in line:
            ids_str = line.split(":")[1].strip().strip('[]')
            stats['unmatched_json_ids'] = [int(id_.strip()) for id_ in ids_str.split(',')] if ids_str else []
    return stats


def write_to_xml(aggregated_data, output_xml):
    root = ET.Element("AggregatedResults")
    for tomo_name, stats in aggregated_data.items():
        tomo_elem = ET.SubElement(root, "Tomogram", name=tomo_name)
        
        masks_elem = ET.SubElement(tomo_elem, "UnmatchedMasks")
        ET.SubElement(masks_elem, "Count").text = str(stats.get('unmatched_masks_count', 0))
        ET.SubElement(masks_elem, "Percentage").text = f"{stats.get('unmatched_masks_percent', 0):.2f}"
        ids_elem = ET.SubElement(masks_elem, "IDs")
        for mask_id in stats.get('unmatched_mask_ids', []):
            ET.SubElement(ids_elem, "ID").text = str(mask_id)
        
        vesicles_elem = ET.SubElement(tomo_elem, "UnmatchedVesicles")
        ET.SubElement(vesicles_elem, "Count").text = str(stats.get('unmatched_vesicles_count', 0))
        ET.SubElement(vesicles_elem, "Percentage").text = f"{stats.get('unmatched_vesicles_percent', 0):.2f}"
        ids_elem = ET.SubElement(vesicles_elem, "IDs")
        for ves_id in stats.get('unmatched_json_ids', []):
            ET.SubElement(ids_elem, "ID").text = str(ves_id)
    
    # Pretty-print the XML and write it to the file
    pretty_xml_string = prettify_xml(root)
    try:
        with open(output_xml, "w") as f:
            f.write(pretty_xml_string)
        print(f"Aggregated results have been written to '{output_xml}' with improved readability.")
    except Exception as e:
        print(f"Error: Unable to write XML file '{output_xml}'. Details: {e}")

def main():
    args = parse_arguments()
    current_path = args.current_path
    broken_json_path = args.broken_json_path
    output_xml = args.output_xml

    broken_data = load_broken_json(broken_json_path)

    if not os.path.isdir(current_path):
        print(f"Error: Current path '{current_path}' is not a directory.")
        sys.exit(1)

    aggregated_data = {}

    # Iterate through all directories in current_path
    for tomo_name in os.listdir(current_path):
        tomo_dir = os.path.join(current_path, tomo_name)
        if not os.path.isdir(tomo_dir):
            continue  # Skip if not a directory

        base_tomo_name = extract_base_tomo_name(tomo_name)

        # Check if segVesicle_heart_broken.json marks tomo_name as broken
        is_broken = broken_data.get(tomo_name, False)
        if is_broken:
            print(f"Skipping '{tomo_name}' because segVesicle_heart_broken.json indicates it is broken.")
            continue

        # Define label_path and json_path
        label_path = os.path.join(current_path, tomo_name, 'ves_seg', f"{base_tomo_name}_label_vesicle.mrc")
        json_path = os.path.join(current_path, tomo_name, 'ves_seg', f"{base_tomo_name}_vesicle.json")

        # Check if label_path and json_path exist
        if not os.path.isfile(label_path):
            print(f"Skipping '{tomo_name}' because label_path '{label_path}' does not exist.")
            continue
        if not os.path.isfile(json_path):
            print(f"Skipping '{tomo_name}' because json_path '{json_path}' does not exist.")
            continue

        print(f"Processing '{tomo_name}'...")

        # Run center_matching.py and capture output
        output = run_center_matching(label_path, json_path)
        if output is None:
            print(f"Skipping '{tomo_name}' due to errors in center_matching.py.")
            continue

        # Parse the output to extract statistics
        stats = parse_center_matching_output(output)
        if stats is None:
            print(f"Skipping '{tomo_name}' due to inability to parse center_matching.py output.")
            continue

        # Store the stats
        aggregated_data[tomo_name] = stats
        sys.stdout.flush()

    if aggregated_data:
        # Write aggregated data to XML
        write_to_xml(aggregated_data, output_xml)
    else:
        print("No data was aggregated. No XML file was created.")

if __name__ == "__main__":
    main()
