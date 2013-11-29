#! /usr/bin/python
# -*- coding:utf-8 -*-
"""
ラベルからいわゆるbig documentを作成する．階層別に分けれる様にするのが理想
"""
__date__='2013/11/30'

import pickle, argparse, re, codecs, os, glob, json, sys;
import return_range;
import numpy;
from nltk.corpus import stopwords;
from nltk import word_tokenize; 
from sklearn.metrics import classification_report;
from sklearn.cross_validation import train_test_split;
from sklearn.svm import LinearSVC;
from sklearn.metrics import confusion_matrix;
from scipy.sparse import lil_matrix
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];

def make_filelist(dir_path):
    file_list=[];
    for root, dirs, files in os.walk(dir_path):
        for f in glob.glob(os.path.join(root, '*')):
            file_list.append(f);
    return file_list;

def extract_leaf_content_for_class_training(class_lebel, target_subtree_map, class_training_stack):
    if not target_subtree_map['child']==[]:
        for child_of_grandchild in target_subtree_map['child']:
            if not child_of_grandchild[1]==None:
                class_training_stack.append((class_lebel,
                                           child_of_grandchild[1].replace(u'\r\n', u'.').strip())); 
    else:
        outline_text=target_subtree_map['content'][1];
        if not outline_text==None:
            class_training_stack.append((class_lebel,
                                         outline_text.replace(u'\r\n', u'.').strip()));

    return class_training_stack;

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

def construct_class_training_1st(parent_node, all_thompson_tree):
    """
    一層目を指定した時に，一層目の各ラベルに属する単語から文書を作って返す
    """
    class_training_stack=[];
    for child_tree_key in all_thompson_tree[parent_node]:
        for grandchild_tree_key in all_thompson_tree[parent_node][child_tree_key]:
            if re.search(ur'[A-Z]_\d+_\d+_\w+', grandchild_tree_key):
                for child_of_grandchild in all_thompson_tree[parent_node][child_tree_key][grandchild_tree_key]:
                    target_subtree_map=all_thompson_tree\
                        [parent_node][child_tree_key][grandchild_tree_key][child_of_grandchild];
                    class_training_stack=\
                        extract_leaf_content_for_class_training(parent_node,
                                                                target_subtree_map,
                                                                     class_training_stack);

            elif re.search(ur'\d+', grandchild_tree_key):
                target_subtree_map=all_thompson_tree[parent_node][child_tree_key][grandchild_tree_key];
                class_training_stack=extract_leaf_content_for_class_training(parent_node,
                                                                             target_subtree_map,
                                                                                  class_training_stack)
    return class_training_stack;

def cleanup_bigdocument_stack(filename, big_document_stack, stop):
    big_document_text=u' '.join(big_document_stack);
    tokens=word_tokenize(big_document_text);
    tokens_s=[t.lower() for t in tokens]
    if stop==True:
        tokens_s=[t for t in tokens if not t in stopwords and not t in symbols];
    print 'Statics of big document';
    print '-'*30;
    print u'filename:{}\nnum. of tokens:{}'.format(filename, len(tokens_s));
    print '-'*30;
    
    return tokens_s;

def cleanup_class_stack(class_training_stack):
    tokens_set_stack=[];
    for tuple_item in class_training_stack:
        label=tuple_item[0];
        tokens=word_tokenize(tuple_item[1]);
        tokens_s=[t.lower() for t in tokens]
        tokens_set_stack.append(tokens_s) 
    return tokens_set_stack;

def make_feature_set(feature_max, feature_map, tokens_set_stack):
    """
    素性関数を作り出す（要はただのmap）
    """
    for token_instance in tokens_set_stack:
        for token in token_instance:
            if token not in feature_map:
                feature_map[token]=feature_max;
                feature_max+=1;
    return feature_max, feature_map;

def construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch):
    training_map={};
    feature_map={};
    feature_max=0;
    num_of_training_instance=0;
    if dutch==True: 
        dir_path='../dutch_folktale_corpus/given_script/translated_big_document/second_layer/' 
        for filepath in make_filelist(dir_path):
            alphabet_label=(os.path.basename(filepath))[0];
            tokens_in_label=word_tokenize(codecs.open(filepath, 'r', 'utf-8').read());
            for token in tokens_in_label:
                if token not in feature_map:
                    feature_map[token]=feature_max;
                    feature_max+=1;
            num_of_training_instance+=1;
            if alphabet_label in training_map:
                training_map[alphabet_label].append(tokens_in_label);
            else:
                training_map[alphabet_label]=[tokens_in_label];
    
    for key_1st in all_thompson_tree:
        parent_node=key_1st;
        class_training_stack=construct_class_training_1st(parent_node, all_thompson_tree);
        tokens_set_stack=cleanup_class_stack(class_training_stack);
        num_of_training_instance+=len(tokens_set_stack);
        feature_max, feature_map=make_feature_set(feature_max, feature_map, tokens_set_stack);
        if key_1st in training_map:
            training_map[key_1st]+=tokens_set_stack;
        else:
            training_map[key_1st]=tokens_set_stack;
    with codecs.open('classifier/feature_map_1st.json', 'w', 'utf-8') as f:
        json.dump(feature_map, f, indent=4, ensure_ascii=False);
    feature_space=len(feature_map);
    #自分で作成したトレーニングモデルがちょっと信用できないので，libsvmも使ってみる
    out_to_libsvm_format(training_map, feature_map);
    
    #ここからtraining用のコードを書き始める
    #training setを構築して，trainingまでをはしらせる
    for label_index, label_name in enumerate(training_map):
        #training用の疎行列(素性次元数＊トレーニング事例数)を先に作成しておく
        training_matrix=lil_matrix((feature_space, num_of_training_instance));
        training_data_label=[];
        instances_in_correct_label=training_map[label_name];
        for col_number, one_instance in enumerate(instances_in_correct_label):
            for feature_token in one_instance:
                #stopwordsの除去するか or not
                if stop==True:
                    if feature_token not in stopwords and feature_token not in symbols: 
                        feature_number=feature_map[feature_token];
                else:
                    feature_number=feature_map[feature_token];
                training_matrix[feature_number, col_number]=1;    
        map(lambda label: training_data_label.append(1), instances_in_correct_label);

        #training_mapからlabel_name以外のキーをすべてよみこんで，素性をベクトル化する
        #label_nameがキーの時の値にだけ正解ラベル１を付与して，
        #残りのキーの時のラベルには不正解ラベル−１を付与する
        for incorrect_label_name in training_map:
            if not incorrect_label_name==label_name:
                instances_in_incorrect_label=training_map[incorrect_label_name];
                for col_number, one_instance in enumerate(instances_in_incorrect_label):
                    for feature_token in one_instance:
                        if stop==True:     
                            if feature_token not in stopwords and feature_token not in symbols:
                                feature_number=feature_map[feature_token];
                        else:
                            feature_number=feature_map[feature_token];
                        training_matrix[feature_number, col_number]=1;    
                map(lambda label: training_data_label.append(0), instances_in_incorrect_label);

        print feature_space, num_of_training_instance, len(training_data_label) 
        tmp=training_matrix.T;
        #この場所でtrainingを実行可能な状態になっているはず
        data_train, data_test, label_train, label_test = train_test_split(tmp,\
                                                                          training_data_label);                
        #分類器にパラメータを与える
        estimator = LinearSVC(C=1.0)
        #与える型はnumpy.ndarrayでないといけない
        #トレーニングデータで学習する
        estimator.fit(tmp, training_data_label)

        #テストデータの予測をする
        label_predict = estimator.predict(data_test)

        print '-'*30;
        print 'label_name:{}'.format(label_name);
        print confusion_matrix(label_test, label_predict)
        target_names=['class0', 'class1'];
        print(classification_report(label_test, label_predict, target_names=target_names))

        #分類器をpickleファイルに出力
        filename='{}_classifier.pickle'.format(label_name);
        with codecs.open('./classifier/1st_layer/'+filename, 'w', 'utf-8') as f:
            pickle.dump(estimator, f);

def out_to_libsvm_format(training_map, feature_map):
    for correct_label_key in training_map:
        outfile=codecs.open('./classifier/libsvm_format/'+correct_label_key, 'w', 'utf-8');
        instances_in_correct_label=training_map[correct_label_key];
        for one_instance in instances_in_correct_label:
            one_instance_stack=[];
            for token in one_instance: 
                feature_number=feature_map[token];
                one_instance_stack.append(str(feature_number)+u':1');
            outfile.write(u'{} {}\n'.format('1', u' '.join(one_instance_stack)));
        for incorrect_label_key in training_map:
            if not correct_label_key==incorrect_label_key:
                instances_in_incorrect_label=training_map[incorrect_label_key];
                for one_instance in instances_in_incorrect_label:
                    one_instance_stack=[];
                    for token in one_instance:
                        feature_number=feature_map[token];
                        one_instance_stack.append(str(feature_number)+u':1');
                    outfile.write(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
        outfile.close();

def main(level, mode, all_thompson_tree, stop, dutch):
    level=int(level);
    level_big_document={};
    #result_stack=return_range.find_sub_tree(input_motif_no, all_thompson_tree) 
    #print 'The non-terminal nodes to reach {} is {}'.format(input_motif_no, result_stack);
    if mode=='big':
        if level==1:
            for key_1st in all_thompson_tree:
                parent_node=key_1st;
                big_document_stack=construct_1st_level(parent_node, all_thompson_tree);
                filename=u'{}_level_{}'.format(parent_node, 1);
                tokens_s=cleanup_bigdocument_stack(filename, big_document_stack, stop);
                with codecs.open('./big_document/'+filename, 'w', 'utf-8') as f:
                    json.dump(tokens_s, f, indent=4, ensure_ascii=False);
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
    
    elif mode=='class':
        training_map={};
        feature_map={};
        feature_max=0;
        num_of_training_instance=0;
        if level==1:
            construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='');
    parser.add_argument('-level', '--level', help='level which you want to construct big doc.', default=1)
    parser.add_argument('-mode', '--mode', help='classification problem(class) or big-document(big)', required=True);
    parser.add_argument('-stop', help='If added, stop words are eliminated from training file', action='store_true');
    parser.add_argument('-dutch', help='If added, document from dutch folktale database is added to training corpus', action='store_true');
    args=parser.parse_args();
    dir_path='./parsed_json/'
    all_thompson_tree=return_range.load_all_thompson_tree(dir_path);
    result_stack=main(args.level, args.mode, all_thompson_tree, args.stop, args.dutch);
