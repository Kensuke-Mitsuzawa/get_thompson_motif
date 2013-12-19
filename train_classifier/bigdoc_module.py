# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:27:31 2013

@author: kensuke-mi
"""
__date__='2013/12/19'
import codecs, json, re, os, glob;
from nltk import tokenize;
from nltk.corpus import stopwords;
from nltk import stem;
lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];

def make_filelist(dir_path):
    file_list=[];
    for root, dirs, files in os.walk(dir_path):
        for f in glob.glob(os.path.join(root, '*')):
            file_list.append(f);
    return file_list;

def extract_leaf_content_for_construct_1st_level(target_subtree_map, big_document_stack):
    if not target_subtree_map['child']==[]:
        for child_of_grandchild in target_subtree_map['child']:
            if not child_of_grandchild[1]==None:
                big_document_stack.append(child_of_grandchild[1].replace(u'\r\n', u'.').strip()); 
    else:
        outline_text=target_subtree_map['content'][1];
        if not outline_text==None:
            big_document_stack.append(outline_text.replace(u'\r\n', u'.').strip());

    return big_document_stack;

def construct_2nd_level(parent_node, sub_thompson_tree):
    """
    ２層目を対象にbig_documentを構築する．つまり１層目の範囲ラベルに対して，そのラベルに属する文書が入る
    ここでparent_nodeは二層目のラベル（つまり範囲）で，sub_thompson_treeはparent_nodeをキーとする要素
    """
    big_document_stack=[];
    for child_key in sub_thompson_tree:
        #child_keyが他の木のキーになってて，さらに下にmapがある時
        if re.search(ur'[A-Z]_\d+_\d+_\w+', child_key):
            for grandchild_key in sub_thompson_tree[child_key]:
                subsubtree_map=sub_thompson_tree[child_key][grandchild_key];
                big_document_stack=\
                    extract_leaf_content_for_construct_1st_level(subsubtree_map, big_document_stack);
        #child_keyが葉要素のキーの時
        elif re.search(ur'\d+', child_key):
            subtree_map=sub_thompson_tree[child_key];
            big_document_stack=extract_leaf_content_for_construct_1st_level(subtree_map, big_document_stack);
    return big_document_stack;        

def cleanup_bigdocument_stack(filename, big_document_stack, stop):
    big_document_text=u' '.join(big_document_stack);
    tokens=tokenize.wordpunct_tokenize(big_document_text);
    tokens_s=[t.lower() for t in tokens]
    if stop==True:
        tokens_s=[t for t in tokens if not t in stopwords and not t in symbols];
    print 'Statics of big document';
    print '-'*30;
    print u'filename:{}\nnum. of tokens:{}'.format(filename, len(tokens_s));
    print '-'*30;
    
    return tokens_s;

def construct_1st_level(parent_node, all_thompson_tree):
    """
    一層目のみを対象にbig-documentを作成する．
    つまり，AからZまでの英数字ラベルに対して，そのラベル中に含まれるすべての語から文書を作成する．
    入力はparent_nodeが頂点ラベル[A-Z]でall_thompson_treeはjsonファイルから読み込んだそのまま
    """
    big_document_stack=[];
    for child_tree_key in all_thompson_tree[parent_node]:
        for grandchild_tree_key in all_thompson_tree[parent_node][child_tree_key]:
            if re.search(ur'[A-Z]_\d+_\d+_\w+', grandchild_tree_key):
                for child_of_grandchild in all_thompson_tree[parent_node][child_tree_key][grandchild_tree_key]:
                    target_subtree_map=all_thompson_tree\
                        [parent_node][child_tree_key][grandchild_tree_key][child_of_grandchild];
                    big_document_stack=\
                        extract_leaf_content_for_construct_1st_level(target_subtree_map, big_document_stack);

            elif re.search(ur'\d+', grandchild_tree_key):
                target_subtree_map=all_thompson_tree[parent_node][child_tree_key][grandchild_tree_key];
                big_document_stack=extract_leaf_content_for_construct_1st_level(target_subtree_map, big_document_stack)
    return big_document_stack;
    
def cleanup_directory(dirpath):
    for f in make_filelist(dirpath):
        os.remove(f);

def create_label1_bigdoc(args, all_thompson_tree):
    dirpath='./big_document/';
    cleanup_directory(dirpath);
    
    #一層目のbigdocumentを生成して書き出し 書き出し先は./big_document/
    if args.thompson==True:
        for key_1st in all_thompson_tree:
            parent_node=key_1st;
            big_document_stack=construct_1st_level(parent_node, all_thompson_tree);
            filename=u'{}_level_{}'.format(parent_node, 1);
            tokens_s=cleanup_bigdocument_stack(filename, big_document_stack, args.stop);
            with codecs.open('./big_document/'+filename, 'w', 'utf-8') as f:
                #json.dump(tokens_s, f, indent=4, ensure_ascii=False);
                #jsonは廃止　通常文書にする
                f.write(u' '.join(tokens_s));

    if args.dutch==True:
        #オランダ語コーパスの疑似文書作成
        big_document_tree={};
        #------------------------------------------------------------ 
        motif_vector=[unichr(i) for i in xrange(65,65+26)];
        motif_vector.remove(u'O'); motif_vector.remove(u'I');
        for m in motif_vector: big_document_tree.setdefault(m, []);
        #------------------------------------------------------------             
        dir_path='../../dutch_folktale_corpus/dutch_folktale_database_translated_kevin_system/translated_train/';                
        for filepath in make_filelist(dir_path):
            tokens_in_label=tokenize.wordpunct_tokenize(codecs.open(filepath, 'r', 'utf-8').read());
            lemmatized_tokens_in_label=[lemmatizer.lemmatize(t.lower()) for t in tokens_in_label];
            if args.stop==True:
                lemmatized_tokens_in_label=[t for t in lemmatized_tokens_in_label if t not in stopwords and t not in symbols];
            tmp_sub=re.sub(ur'([A-Z]_+)\d.+', ur'\1', os.path.basename(filepath.upper()));
            for label in (tmp_sub).split(u'_')[:-1]:
                if label in big_document_tree:
                    big_document_tree[label]+=lemmatized_tokens_in_label;
                else:
                    big_document_tree[label]=lemmatized_tokens_in_label;
        #------------------------------------------------------------
        #通常文書として書き出し
        print 'writing tokens in dutch folktale database';
        for label in big_document_tree:
            with codecs.open('./big_document/'+label+'_level_1', 'a', 'utf-8') as f:
                doc_in_label=big_document_tree[label];
                f.write(u' '.join(doc_in_label));
    
def big_doc_main(all_thompson_tree, args):
    level=args.level;
    stop=args.stop;
    if level==1:
        create_label1_bigdoc(args, all_thompson_tree);

    elif level==2:
        for key_1st in all_thompson_tree:
            for key_2nd in all_thompson_tree[key_1st]:
                parent_node=key_2nd;
                sub_thompson_tree=all_thompson_tree[key_1st][key_2nd];
                big_document_stack=construct_2nd_level(parent_node, sub_thompson_tree); 
                parent_node=re.sub(ur'([A-Z]_\d+_\d+).+', r'\1', parent_node);
                filename=u'{}_level_{}'.format(parent_node, 2);
                tokens_s=cleanup_bigdocument_stack(filename, big_document_stack, stop);
                with codecs.open('./big_document/'+filename, 'w', 'utf-8') as f:
                    json.dump(tokens_s, f, indent=4, ensure_ascii=False);
    #TODO 必要に応じて３層目を作成する
