# Create Pre-train dataset from BioLiP
import pandas as pd
import numpy as np
import os
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select, MMCIFIO
from tqdm import tqdm
from collections import defaultdict
from data.protein.residue_constants import restype_3to1
import concurrent.futures
import itertools as it
from operator import length_hint
import threading
rna_residues = ['A', 'G', 'C', 'U']
protein_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
convert_dict = {'mM': 1e-3, 'uM': 1e-6, 'nM': 1e-9, 'pM': 1e-12, 'fM': 1e-15}


def search_biolip(item, df_biolip):
    sub_df = df_biolip[df_biolip['PDB ID']==item.lower()]
    if len(sub_df) == 0:
        return False, []
    else:
        pro_rna_dict = defaultdict(list)
        rna_pro_dict = defaultdict(list)
        prot_chains = []
        prot_bind_units = []
        rna_bind_units = []
        for _, row in sub_df.iterrows():
            prot_chain = row['Receptor chain']
            rna_chain = row['Ligand chain']
            pro_rna_dict[prot_chain].append(rna_chain)
            rna_pro_dict[rna_chain].append(prot_chain)
            prot_chains.append(prot_chain)
        start = prot_chains[0]
        prot_bind_units.append(start)
        search = 1
        binding_pairs = []
        while search == 1:
            search = 0
            # print("Hi")
            for prot in prot_bind_units:
                for rna in pro_rna_dict[prot]:
                    if rna not in rna_bind_units:
                        rna_bind_units.append(rna)
                        binding_pairs.append((prot, rna))
            for rna in rna_bind_units:
                for prot in rna_pro_dict[rna]:
                    if prot not in prot_bind_units:
                        prot_bind_units.append(prot)
                        binding_pairs.append((prot, rna))
                        search = 1

        # prot_chains = ','.join(prot_bind_units)
        # rna_chains = ','.join(rna_bind_units)
        pairs = list(set(binding_pairs))
        return True, pairs       

class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = chains

    def accept_chain(self, chain):
        return chain.get_full_id()[2] in self.chains

def save_structure(structure, output_file, chains):
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_file, ChainSelect(chains))


def process_rna(resname):
    if resname in rna_residues:
        return resname
    if resname == 'T':
        return 'U'
    else:
        return '_'

def cal_from_structure(item, dataset_root):
    parser = MMCIFParser(QUIET=True)
    # print(os.path.join(dataset_root, item))
    structure = parser.get_structure('', os.path.join(dataset_root, item))
    model = structure[0]
    chain_dict = {}
    # print("Item:", item)
    for chain in model:
        full_id = chain.get_full_id()
        chain_id = full_id[2]
        # print("Full id", chain_id)
        found = 0
        for residue in chain:
            if residue.get_resname() in protein_residues:
                residues = "".join([chain_id+ ":"] + [restype_3to1.get(residue.get_resname(), 'X') for residue in chain if (residue.get_resname() != 'HOH' and residue.id[0] == ' ' and residue.id[2] == ' ')])
                # print(residues)
                prot_seq = residues
                chain_dict[chain_id] = prot_seq
                found = 1
                break
            elif residue.get_resname() in rna_residues:
                residues = "".join([chain_id + ":"] + [process_rna(residue.get_resname())  for residue in chain if (residue.get_resname() != 'HOH' and residue.id[0] == ' ' and residue.id[2] == ' ')])
                # print(residues)
                # print(residues)
                na_seq = residues
                chain_dict[chain_id] = na_seq
                found = 1
                break
        if found == 0:
            print("Chain ", item, ": ", chain_id, "Not hit!")
    return structure, chain_dict

def process_biolip(item, df_biolip, final_results):
    # print("Parameters:", item)
    # valid, results = search_biolip(item, df_biolip)
    sub_df = df_biolip[df_biolip['PDB ID']==item.lower()]
    pairs = []
    for _, row in sub_df.iterrows():
        prot_chain = row['Receptor chain']
        rna_chain = row['Ligand chain']
        pairs.append([prot_chain, rna_chain])
    # if not valid:
    #     print("Weird!")
    item = item.upper()
    item_dict = {        
                'PDB': [],
                'Protein chains': [],
                'RNA chains': [],
                'Protein sequences': [],
                'RNA sequences': []
                }
    output_dir = './datasets/BioLiP2/PDBs'
    structure, chain_dict = cal_from_structure(item+'.cif', '/public/home/HR/PIXberts/protein_rna_complexes')
    # with lock:
    count = 0
    for pair in pairs:
        prot, rna = pair[0], pair[1]
        if prot not in chain_dict or rna not in chain_dict:
            print("Not found", item, prot, rna)
            continue
        # print(chain_dict[prot], chain_dict[rna])
        if len(chain_dict[prot]) <= 1000 and len(chain_dict[rna]) <= 250:
            item_dict['PDB'].append(item)
            item_dict['Protein chains'].append(prot)
            item_dict['RNA chains'].append(rna)
            item_dict['Protein sequences'].append(chain_dict[prot])
            item_dict['RNA sequences'].append(chain_dict[rna])
            selected_chains = [prot, rna]
            item_name = '{}_{}_{}.cif'.format(item, prot, rna)
            count += 1
            if os.path.exists(os.path.join(output_dir, item_name)):
                continue
            # save_structure(structure, os.path.join(output_dir, item_name),selected_chains)
        # else:
        #     print("Too long:", item, prot, rna)
    final_results.append(item_dict)
    return count


if __name__ == '__main__':
    df_biolip = pd.read_csv('./datasets/BioLiP2/BioLiP.csv')
    items = list(set(df_biolip['PDB ID']))

    with open('/public/home/HR/PIXberts/list_of_pdbs.txt', 'r') as f:
        protein_rna = f.read()
        protein_rna_list = protein_rna.split(',')
    items = [item for item in items if item.upper() in protein_rna_list]
    print("Number of Protein-RNA complexes:", len(items))
    # print(items)

    out_dict = {
        'PDB': [],
        'Protein chains': [],
        'RNA chains': [],
        'Protein sequences': [],
        'RNA sequences': []
    }
    final_results = []
    threads = []
    # lock = threading.Lock()
    sample = 0
    t_bar = tqdm(total=len(items), desc='Processing items') 
    for item in items:
        # thread = threading.Thread(target=process_biolip, args=(item, df_biolip, final_results, lock))
        # threads.append(thread)
        # thread.start()
        new_samples = process_biolip(item, df_biolip, final_results)
        sample += new_samples
        t_bar.update(1)
        t_bar.set_description("Total processed: {}".format(sample))
        t_bar.refresh()

        
    # for thread in threads:
    #     thread.join()
        
    keys = out_dict.keys()
    print("Done! Combining results...")
    for key in tqdm(keys):
        for result in final_results:
            out_dict[key] += result[key]

    out_df = pd.DataFrame(out_dict)
    print("Number of entries:", len(out_df))
    out_df.to_csv('./datasets/BioLiP2/for_pretrain_fixbug.csv', index=False)