'''
    @Author : Rashmi Arora
'''    

import numpy as np 
import pandas as pd
import itertools as iter
from collections import Counter
import os
import sys, getopt

#Candidate Generation for Level-2
def level2_candidate_gen(L,SDC) :
    C2 = {'F_Itemset' : []}
    dict_value = []
    init_indices = L[L["Sup_Count"].values >= L["MIS"].values].index.values    
    for init_index in init_indices:   
        df_from_init_index = L[np.where(L.index.values == init_index)[0][0] + 1 :]
        L2 = df_from_init_index[np.logical_and(df_from_init_index["Sup_Count"] >= 
                                 L.at[init_index, "MIS"],
                                 (df_from_init_index["Sup_Count"] - 
                                  L.at[init_index, "Sup_Count"]).abs() <= SDC)].index.values
        for h in L2:            
            dict_value.append([init_index, h])
    C2['F_Itemset'] = dict_value
    return C2
    
#Candidate Generation for Levels > 2
def MScandidate_gen(F_prev, L, SDC):
    C = {'F_Itemset': []}    
    F_prev = sorted(F_prev, key = lambda x: int(x[-1]) if (x[-1].isdigit() == True) else x[-1]) 
    f = list(filter(lambda x : (int(x[0][-1]) < int(x[1][-1]) if (x[0][-1].isdigit() == True) else x[0][-1] < x[1][-1]) and 
                    (x[0][:-1] == x[1][:-1]) and (np.absolute(L.at[x[0][-1], "Sup_Count"] 
                    - L.at[x[1][-1], "Sup_Count"]) <= SDC), iter.combinations(F_prev,2)))
    Ck = [[*item[0], item[1][-1]] for item in f]
    temp = [c for c in Ck if (any((c[0] in list(s) or (L.at[c[1], "MIS"] == L.at[c[0], "MIS"])) and 
            not list(s) in F_prev for s in list(iter.combinations(c,len(c) - 1))))]
    Ck = [c for c in Ck if c not in temp]
    C['F_Itemset'] = Ck
    return C
    
def MS_Apriori(t_input, parameters):
    itemset = {}
    input_list = []
    temp_list = []
    temp_str = ""
    sup_count = {}
    total_t = 0
    result = pd.DataFrame(columns = ['F_value', 'F_Itemset', 'Sup_Count', 'tailcount' ])
    
    #extracting parameters from parameter-file.txt
    with open(parameters, 'r') as file:
        must_have = []
        cannot_have = []
        for line in file:
            if "mis" in line.lower():
                key = line[line.find('(') + 1 : line.find(')')]
                value = [float(line[line.find('=') + 2 : ].strip('\n')), 0.0]
                itemset[key] = value
            elif "sdc" in line.lower():
                SDC = float(line[line.find('=') + 2 : ].strip('\n') )           
            elif "cannot" in line.lower():
                cannot_have = line[line.find(':') + 2 : ].strip('\n')
                cannot_have = cannot_have.replace("}, ","").split("{")[1:]
                cannot_have = [list(map(str, s.replace('}','').split(', '))) for s in cannot_have]
            elif "must" in line.lower():            
                must_have = line[line.find(':') + 2 : ].strip('\n')
                must_have = list(must_have.split(" or "))           
        
    #extracting input from input-file.txt
    with open(t_input, 'r') as file:
        input_list = [set(line.strip('\n')[1:-1].split(', ')) for line in file if line.strip()]
    total_t = len(input_list)
    Sup_Counter = Counter(iter.chain.from_iterable(input_list))
    for key in Sup_Counter:
        itemset[key][1] = Sup_Counter[key] / total_t  
      
    #construct a DataFrame using dict itemset
    df_itemset = pd.DataFrame(itemset)
    df_itemset = df_itemset.transpose()
    df_itemset.columns = ['MIS', 'Sup_Count']
    
    #sort DataFrame based on MIS value
    df_sorted = df_itemset.sort_values('MIS')
    
    #compute L : init-pass(2)
    F1 = pd.DataFrame()
    L = pd.DataFrame()    
    init_indices = df_sorted[df_sorted["Sup_Count"].values >= df_sorted["MIS"].values].index.values
    if(init_indices.any()):
        init_index = init_indices[0]        
        df_from_init_index = df_sorted[np.where(df_sorted.index.values == init_index)[0][0] :]
        init_MIS = df_from_init_index.at[init_index, "MIS"]        
        L = df_from_init_index[df_from_init_index["Sup_Count"].values >= init_MIS]
        F1 = L[L["Sup_Count"] >= L["MIS"]]
        
    with open('output.txt', 'w+') as out_file:
        out_file.write("Frequent 1-itemsets\n")
        if(not F1.empty):
            if(any(must_have)):
                R = [c for c in F1.index.values if any(c==x for x in must_have)]
            else:
                R = F1.index.values
            for index in R:
                out_file.write("\n\t"+ str(Sup_Counter[index]) + " : " + str({index}))
            out_file.write("\n\n\tTotal number of frequent 1-itemsets = " + str(len(R)))
        else:
            out_file.write("\n\tNone")
            out_file.write("\n\n\tTotal number of frequent 1-itemsets = " + str(0))
            return 
    
    #computing rest of the F
    k=1
    while(True):
        if k==1 :
            cand_dict = level2_candidate_gen(L,SDC) 
        else:
            cand_dict = MScandidate_gen(list(F['F_Itemset']),L, SDC)
        if(cand_dict["F_Itemset"]):
            cand_df = pd.DataFrame(cand_dict)
            cand_df["First_Element"] = cand_df['F_Itemset'].apply(lambda col : col[0])
            cand_df["Sup_Count"] = 0
            cand_df["tailcount"] = 0
            
            for x in cand_df["F_Itemset"].values:
                index_x = list(cand_df.F_Itemset).index(x)
                tail = [set(x[1:]) <= t for t in input_list]
                cand_df.loc[index_x, "tailcount"] = sum(tail)           
                cand_df.loc[index_x, "Sup_Count"] = sum([set(x) <= t for t in iter.compress(input_list,tail)])
            cand_df["Sup_Count"] = cand_df["Sup_Count"] / total_t
            F = cand_df[cand_df["Sup_Count"] >= cand_df["First_Element"].apply(lambda x : df_sorted.at[x,"MIS"])]
            
            if(not F.empty):
                result = result.append(F[['F_Itemset','Sup_Count','tailcount']], ignore_index = True)
                result[['F_value']] = result[['F_value']].fillna(value=(k+1))
                k+=1
            else:
                break           
        else:
            break
    
    result['Sup_Count'] = result['Sup_Count'].values * total_t
    
    if(not result.empty):
        if(any(cannot_have)):
            A = result[result['F_Itemset'].apply(lambda c : True if all( not set(x) <= set(c) for x in cannot_have) else False)]
        else:
            A = result
        if(not A.empty):
            if(any(must_have)):
                B = A[A['F_Itemset'].apply(lambda c : True if any({x} <= set(c) for x in must_have) else False)]
            else :
                B = A
                
        with open('output.txt', 'a') as out_file:
            if(not(B.empty)):
                for i in range(2, int(B['F_value'].max()) + 1):
                    out_file.write("\n\nFrequent " + str(i) + "-itemsets\n")
                    group = B[B['F_value'] == i]
                    for index in group.index.values:
                        out_file.write("\n\t"+ str(int(group.loc[index,"Sup_Count"])) + " : " + str(group.loc[index,"F_Itemset"]).replace('[','{').replace(']','}'))
                        out_file.write("\nTailcount = " + str(group.loc[index,"tailcount"]))
                    out_file.write("\n\n\tTotal number of frequent " + str(i) + "-itemsets = " + str(group["F_Itemset"].size))

#calling MS_Apriori
if __name__ == "__main__":
    argv = sys.argv[1:]
    input_file = ""
    output_file = ""
    param_file = ""
    try:
        opts, args = getopt.getopt(argv,"i:o:p:",["ifile=","pfile="])
    except getopt.GetoptError:
        print("Invalid Command/Arguments")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
             input_file = arg
        elif opt in ("-p", "--pfile"):
             param_file = arg
    cur_dir = os.getcwd()
    input_file_path = os.path.join(cur_dir, input_file)
    param_file_path = os.path.join(cur_dir, param_file)
    MS_Apriori(input_file_path, param_file_path)
    print("Success! The results can be seen in file named output.txt in the same directory as other files. ")

