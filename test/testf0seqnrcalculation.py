"""
Example usage:

python3 testf0seqnrcalculation.py 548659 \
        ../testdata/sample_data_548659/L548659.parset \
        ../testdata/sample_data_548659/file-sizes.txt \
        ../testdata/sample_data_548659/f0seqnr-sizes.txt 
"""

import os
import sys

sys.path.append("../scripts") 
import create_html




def test_main(in_sas_id, in_parset_path, in_file_sizes_path, in_f0seqnr_sizes_path):
    result = True
    
    parset = create_html.parset_summary(in_sas_id, in_parset_path)
    print("Relevant parset details:")
    print("clock_mhz: ", parset['clock_mhz'])
    print("start-time: ", parset['start_time'])
    print("stop-time: ", parset['stop_time'])
    print("block_size: ", parset['block_size'])
    print("nr_integrations_per_block: ", parset['nr_integrations_per_block'])
    print("nr_blocks_per_integration: ", parset['nr_blocks_per_integration'])
    print("nr_integration_periods: ", parset['nr_integration_periods'])    
    print("Correlator locations: ", "\n".join(parset['correlator_locations']))
    print("Beamformer locations: ", "\n".join(parset['beamformer_locations']))
    
    
    print("\nContent of [file_sizes_dict]:")
    file_sizes_dict = create_html.parse_file_sizes(in_file_sizes_path)
    for data_product_folder, (_, _, file_size_in_mb) in file_sizes_dict.items():    
        print(data_product_folder, " (", file_size_in_mb, "MB)")       
    
    
    print("\nContent of [analysed_file_sizes_dict]:")
    analysed_file_sizes_dict  = create_html.file_size_analysis(parset, file_sizes_dict)
    highest_file_size_mb = analysed_file_sizes_dict['max_ms_size_mb'].max()
    print("max_ms_size_mb: ", analysed_file_sizes_dict['max_ms_size_mb'])
    print("max_raw_size_mb: ", analysed_file_sizes_dict['max_raw_size_mb'])
    print("missing_data_sets: ", analysed_file_sizes_dict['missing_data_sets'])
    print("odd_sized_data_sets: ", analysed_file_sizes_dict['odd_sized_data_sets'])
    print("percentage_complete: ", analysed_file_sizes_dict['percentage_complete']) 
    
    print("\nContent of [f0seqnr_sizes_dict]:")
    f0seqnr_sizes_dict = create_html.parse_file_sizes(in_f0seqnr_sizes_path, os.path.dirname)
    for data_product_folder, (_, _, nr_integration_periods_in_file) in f0seqnr_sizes_dict.items():        
        print(data_product_folder, " (", nr_integration_periods_in_file, ")")        
    
    print("\nContent of [completeness_dict]:")
    completeness_dict = create_html.f0seqnr_size_analysis(parset, f0seqnr_sizes_dict)
    for data_product_folder, completeness_value in completeness_dict.items():
        print(data_product_folder, " (", completeness_value, ")")
        
    print("\nTotal average completeness: ", create_html.calculate_average(completeness_dict.values()))
    
    print("\nIncomplete datasets according to original method (odd_sized_data_sets (=) abs(float(data_size_mb)/float(max_ms_size_mb) -1.0) > 0.01):")
    for (name, size) in sorted(analysed_file_sizes_dict['odd_sized_data_sets']):
        print("Dataset: ", name, " Size: ", size, "MB")
    
    print("\nIncomplete datasets according to f0seqnr method (completeness_value < 100):")
    for data_product_folder, completeness_value in completeness_dict.items():
        if completeness_value < 100:
            print("Dataset: ", data_product_folder, " Completeness: %0.1f%%" % completeness_value)
            
    print("\nIncomplete datasets based on relative (%rMB) file size:" % highest_file_size_mb)
    for data_product_folder, (_, _, file_size_in_mb) in file_sizes_dict.items():  
        if file_size_in_mb < highest_file_size_mb:
            print("Dataset: ", data_product_folder, " Size: ", file_size_in_mb, "MB ", "(%0.1f%%)" % (100*file_size_in_mb/highest_file_size_mb))
            
            
    print('\n'.join(['%s: %dMB (%0.f%%)' % (name, size, completeness_dict.get(name, -1)) for (name, size) in sorted(analysed_file_sizes_dict['odd_sized_data_sets'])]))
    
    return result
    
    
def parse_arguments(argv):  
    sas_id = int(argv[1])
    parset_path = argv[2]
    file_sizes_path = argv[3]
    f0seqnr_sizes_file_path = argv[4]
    return sas_id, parset_path, file_sizes_path, f0seqnr_sizes_file_path


if __name__ == '__main__':
    print("Called as: %s\n" % (" ".join(sys.argv)))
    if len(sys.argv) == 5:
        sas_id, parset_path, file_sizes_path, f0seqnr_sizes_file_path = parse_arguments(sys.argv)
        
        if test_main(sas_id, parset_path, file_sizes_path, f0seqnr_sizes_file_path):
            print("Test successful")
        else:
            print("Test unsuccessful")
    else:
        print ("Usage:\ntestf0seqnrcalculation [SAS ID] [parset file_path] [file_sizes_path] [f0seqnr_sizes_file_path]")
