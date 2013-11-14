#! /usr/bin/python
# -*- coding:utf-8 -*-
"""
入力されたモチーフノードに到達するまでの中間ノードを調べる．粒度を変えてモチーフ予測の精度を行うことが可能
ARGS: 調べたいモチーフ番号([A-Z]\d+ の形式で入力すること　例：F913)
"""

import re, codecs, os, glob, json, sys;

def unify_json_files(dir_path):
    all_thompson_tree={};
    for root, dirs, file in os.walk(dir_path):
        for f in glob.glob(os.path.join(root, '*')):
            filename=os.path.basename(f);
            prefix=(filename.split(u'.')[0]).upper();
            with codecs.open(f, 'r', 'utf-8') as f:
                thompson_tree=json.load(f);
                all_thompson_tree.setdefault(prefix, thompson_tree);
    with codecs.open(dir_path+'/unified_json', 'w', 'utf-8') as f:
        json.dump(all_thompson_tree, f, indent=4, ensure_ascii=False);
    return all_thompson_tree;

def load_all_thompson_tree():
    with codecs.open('./parsed_json/unified_json', 'r', 'utf-8') as f:
        all_thompson_tree=json.load(f);
    return all_thompson_tree;

def find_sub_tree(input_motif_no, all_thompson_tree):
    result_stack=[];
    prefix=input_motif_no[0];    
    query_number=int(input_motif_no[1:]);
    subtree_map=all_thompson_tree[prefix];
    for keyname_1level in subtree_map:
        start_range_1level=int(keyname_1level.split(u'_')[1]);
        end_range_1level=int(keyname_1level.split(u'_')[2]);
        if query_number>=start_range_1level and query_number<=end_range_1level:
            if str(query_number) in subtree_map[keyname_1level]:
                print keyname_1level;
                result_stack.append(keyname_1level);
            #さらに下の階層を掘り下げ
            for keyname_2level in subtree_map[keyname_1level]:
                #もし，キーが下の形をしていたら，まださらに階層があるということ
                if re.search(ur'\w_\d+_\d+_\w+', keyname_2level):
                    start_range_2level=int(keyname_2level.split(u'_')[1]); 
                    end_range_2level=int(keyname_2level.split(u'_')[2]);
                    if query_number>=start_range_2level and query_number<=end_range_2level:
                        print keyname_2level;
                        result_stack.append(keyname_2level);
    return result_stack; 
def main(input_motif_no):
    dir_path='./parsed_json/'
    #all_thompson_tree=unify_json_files(dir_path);
    all_thompson_tree=load_all_thompson_tree();
    result_stack=find_sub_tree(input_motif_no, all_thompson_tree) 
    print 'The non-terminal nodes to reach {} is {}'.format(input_motif_no, result_stack);
if __name__=='__main__':
    input_motif_no='F913';
    #input_motif_no=sys.argv[1].decode('utf-8');
    main(input_motif_no);
