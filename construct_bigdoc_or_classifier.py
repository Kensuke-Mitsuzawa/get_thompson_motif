#! /usr/bin/python
# -*- coding:utf-8 -*-
"""
ラベルからいわゆるbig documentを作成する．階層別に分けれる様にするのが理想
"""
__date__='2013/12/02'
libsvm_wrapper_path='/home/kensuke-mi/opt/libsvm-3.17/python/';
#TODO 開発モードを入れる（トレーニング事例のインデックスをめちゃ少なくするとか）
import subprocess, random, pickle, argparse, re, codecs, os, glob, json, sys;
sys.path.append(libsvm_wrapper_path);
from liblinearutil import *;
from svmutil import *;
import return_range, tf_idf;
import numpy;
from nltk.corpus import stopwords;
from nltk import word_tokenize; 
from sklearn import svm;
from sklearn.metrics import classification_report;
from sklearn.cross_validation import train_test_split;
from sklearn.svm import LinearSVC;
from sklearn.metrics import confusion_matrix;
from scipy.sparse import lil_matrix
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];
#option parameter
put_weight_constraint=True;
under_sampling=False;
level=1;
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
    一層目を指定した時に，一層目の各ラベルに属する単語から訓練事例を作って返す
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

def preprocess_particular_case(sentence):
    if re.search(ur'\w\.\w', sentence):
        sentence=re.sub(ur'(\w)\.(\w)', ur'\1 \2', sentence);
        return sentence; 
    else:
        return sentence;

def cleanup_class_stack(class_training_stack, stop):
    tokens_set_stack=[];
    for tuple_item in class_training_stack:
        label=tuple_item[0];
        cleaned_sentence=preprocess_particular_case(tuple_item[1]);
        tokens=word_tokenize(cleaned_sentence);
        tokens_s=[t.lower() for t in tokens]
        if stop==True:
            tokens_set_stack.append([t for t in tokens if t not in stopwords and t not in symbols]);
        else:
            tokens_set_stack.append(tokens_s) 
    return tokens_set_stack;

def make_feature_set(feature_map, label_name, tokens_set_stack, feature_mode, stop):
    """
    素性関数を作り出す（要はただのmap）
    """
    if feature_mode=='hard':
        for token_instance in tokens_set_stack:
            for token in token_instance:
                hard_cluster_feature=u'hard_{}_{}_unigram'.format(label_name, token);
                character_feature=hard_cluster_feature;
                if stop==True and token not in stopwords and token not in symbols:
                    if token in feature_map and character_feature not in feature_map[token]:
                        feature_map[token].append(character_feature);
                    else:
                        feature_map[token]=[character_feature];
                elif stop==False:
                    if token in feature_map and character_feature not in feature_map[token]:
                        feature_map[token].append(character_feature);
                    else:
                        feature_map[token]=[character_feature];
    #allはすでにsoftの素性が存在しているときに付与
    elif feature_mode=='all':
        for token_instance in tokens_set_stack:
            for token in token_instance:
                normal_cluster_feature=u'all_{}_unigram'.format(token);
                character_feature=normal_cluster_feature; 
                if stop==True and token not in stopwords and token not in symbols:
                    if token in feature_map and character_feature not in feature_map[token]:
                        feature_map[token].append(character_feature);
                    else:
                        feature_map[token]=[character_feature];
                elif stop==False:
                    if token in feature_map and character_feature not in feature_map[token]:
                        feature_map[token].append(character_feature);
                    else:
                        feature_map[token]=[character_feature];
    return feature_map;
def make_soft_char_feature(training_map, feature_map, stop):
    """
    Dutch folktale corpusの中でマルチラベルのtokenを見つけ出す．
    見つかったマルチラベルなtokenはword_union_mapに保存して，その後，soft_から始まる素性を作成して，feature_map_characterに登録する．
    """ 
    word_cap_map={}; 
    for label_1st in training_map:
        token_1st=[]; token_2nd=[];
        for token in training_map[label_1st]: [token_1st.append(t) for t in token];
        tokens_set_1st=set(token_1st);
        for label_2nd in training_map:
            if label_1st!=label_2nd:
                for token in training_map[label_2nd]: [token_2nd.append(t) for t in token];
                tokens_set_2nd=set(token_2nd);
                cap_2_set=tokens_set_1st.intersection(tokens_set_2nd);
                for cap_token in cap_2_set:
                    if cap_token not in word_cap_map:
                        word_cap_map[cap_token]=[label_1st, label_2nd];
                    else:
                        if label_2nd not in word_cap_map[cap_token]:
                            word_cap_map[cap_token].append(label_2nd);
    #ちょっと無駄な処理になるが，もう一度for文を回して，capでない素性をつくりだす
    #ある特定の文書にしか登場していない素性
    for label in training_map:
        for tokens in training_map[label]:
            for token in tokens:
                if token not in word_cap_map:
                    if stop==True and token not in stopwords and token not in symbols:
                        feature_map[token]=[u'soft_{}_{}_unigram'.format(label, token)];
                    elif stop==False:
                        feature_map[token]=[u'soft_{}_{}_unigram'.format(label, token)];
    for key in word_cap_map:
        L=word_cap_map[key];
        L.sort();
        soft_feature=u'soft_{}_{}_unigram'.format(u'_'.join(L), key);
        if stop==True and token not in stopwords and token not in symbols:
            if key in feature_map:
                feature_map[key].append(soft_feature);
            else:
                feature_map[key]=[soft_feature];
        elif stop==False:
            if key in feature_map:
                feature_map[key].append(soft_feature);
            else:
                feature_map[key]=[soft_feature];
    return feature_map;

def make_numerical_feature(feature_map_character):
    feature_map_numeric={};
    feature_num_max=1;
    for token_key in feature_map_character:
        for feature in feature_map_character[token_key]:
            feature_map_numeric[feature]=feature_num_max; 
            feature_num_max+=1;
    return feature_map_numeric;

def construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch, thompson, tfidf, exno):
    exno=str(exno);
    training_map={};
    tfidf_score_map={};
    feature_map_character={};
    num_of_training_instance=0;
    #============================================================ 
    #オランダ語コーパスとトンプソン木をフラッグによって，訓練に使うかどうかを分岐
    if dutch==True:
        dir_path='../dutch_folktale_corpus/given_script/translated_big_document/leaf_layer/' 
        #------------------------------------------------------------
        #文書を全部よみこんで，training_mapの下に登録する．前処理みたいなもん
        for filepath in make_filelist(dir_path):
            if level==1:
                alphabet_label_list=(os.path.basename(filepath)).split('_')[:-1];
            elif level==2:
                alphabet_label=(os.path.basename(filepath))[0];
            tokens_in_label=word_tokenize(codecs.open(filepath, 'r', 'utf-8').read());
            if stop==True:
                tokens_in_label=[t for t in tokens_in_label if t not in stopwords and t not in symbols];
            if level==1:
                for alphabet_label in alphabet_label_list:
                    if alphabet_label in training_map:
                        training_map[alphabet_label].append(tokens_in_label);
                    else:
                        training_map[alphabet_label]=[tokens_in_label];
            elif level==2:
                if alphabet_label in training_map:
                    training_map[alphabet_label].append(tokens_in_label);
                else:
                    training_map[alphabet_label]=[tokens_in_label];
        #------------------------------------------------------------ 
        #training_mapへの登録が全部おわってから，素性抽出を行う 
        #easy domain adaptation用にここで工夫ができるはず
        if tfidf==False:
            #A~Zのラベル間でcapな単語を求めだす
            #全ラベル間でcapな単語を作成して，{token}:'capなラベルをアンダースコア接続で表現'
            #TODO この関数にミスがあると思う．複数のラベルが取得できていない
            feature_map_character=make_soft_char_feature(training_map, feature_map_character, stop);
            doc_token=[];
            for label in training_map:
                doc=training_map[label];
                doc_token+=doc; 
            feature_map_character=make_feature_set(feature_map_character, None, doc_token, 'all', stop);
        #------------------------------------------------------------ 
        elif tfidf==True:
            docs=[];
            for label in training_map:
                doc=training_map[label];
                docs+=doc;
            tfidf_score_map=tf_idf.tf_idf_test(docs);
            for token_key in tfidf_score_map:
                if token_key not in feature_map_character:
                    feature_map_character[token_key]=[u'all_{}_{}_tfidf'.format(token_key, 
                                                                           tfidf_score_map[token_key])]
                else:
                    feature_map_character[token_key].append(u'all_{}_{}_tfidf'.\
                                                            format(token_key,
                                                                   tfidf_score_map[token_key]));
        
        for alphabet_label in training_map:
            print u'The num. of training instance for {} in dutch corpus is {}'.format(alphabet_label, len(training_map[alphabet_label]));
        print u'-'*30;
    #============================================================ 
    #Thompsonのインデックスツリーを訓練データに加える
    if thompson==True: 
        for key_1st in all_thompson_tree:
            parent_node=key_1st;
            class_training_stack=construct_class_training_1st(parent_node, all_thompson_tree);
            tokens_set_stack=cleanup_class_stack(class_training_stack, stop);
            print u'-'*30;
            print u'Training instances for {} from thompson tree:{}'.format(key_1st,len(tokens_set_stack));
            num_of_training_instance+=len(tokens_set_stack);
            #------------------------------------------------------------ 
            #作成した文書ごとのtokenをtrainingファイルを管理するmapに追加
            #TFIDFがTrueだろうが，Falseだろうが関係なく，ここは実行される
            if key_1st in training_map:
                training_map[key_1st]+=tokens_set_stack;
            else:
                training_map[key_1st]=tokens_set_stack;
        #------------------------------------------------------------ 
        #素性をunigram素性にする
        #easy domain adaptation用にここで工夫ができるはず
        if tfidf==False:
            for label in training_map:
                tokens_set_stack=training_map[label];
                #文字情報の素性関数を作成する
                feature_map_character=make_feature_set(feature_map_character,
                                                       label, tokens_set_stack, 'hard', stop);
                feature_map_character=make_feature_set(feature_map_character,
                                                       label, tokens_set_stack, 'all', stop);
        #------------------------------------------------------------ 
        #training_mapに登録がすべて終わってから素性抽出
        elif tfidf==True:
            docs=[];
            for label in training_map:
                doc=training_map[label];
                docs+=doc;
            tfidf_score_map=tf_idf.tf_idf_test(docs);
            for token_key in tfidf_score_map:
                if token_key not in feature_map_character: 
                    feature_map_character[token_key]=[u'all_{}_{}_tfidf'.format(token_key,
                                                                            tfidf_score_map[token_key])];
                else:
                    feature_map_character[token_key].append(u'all_{}_{}_tfidf'.format(token_key,
                                                                            tfidf_score_map[token_key]));
    #============================================================  
    #作成した素性辞書をjsonに出力(TFIDF)が空の時は空の辞書が出力される
    with codecs.open('classifier/tfidf_word_weight.json.'+exno, 'w', 'utf-8') as f:
        json.dump(tfidf_score_map, f, indent=4, ensure_ascii=False);
    with codecs.open('classifier/feature_map_character_1st.json.'+exno, 'w', 'utf-8') as f:
        json.dump(feature_map_character, f, indent=4, ensure_ascii=False);
    #ここで文字情報の素性関数を数字情報の素性関数に変換する
    feature_map_numeric=make_numerical_feature(feature_map_character);
    with codecs.open('classifier/feature_map_numeric_1st.json.'+exno, 'w', 'utf-8') as f:
        json.dump(feature_map_numeric, f, indent=4, ensure_ascii=False);

    feature_space=len(feature_map_numeric);
    print u'The number of feature is {}'.format(feature_space)
    #自分で作成したトレーニングモデルがちょっと信用できないので，libsvmも使ってみる
    out_to_libsvm_format(training_map, feature_map_numeric, feature_map_character, tfidf, exno);
   
    """
    num_of_correct_training_instance=0;
    num_of_incorrect_training_instance=0;
    #ここからtraining用のコードを書き始める
    #training setを構築して，trainingまでをはしらせる
    for label_index, label_name in enumerate(training_map):
        #training用の疎行列(素性次元数＊トレーニング事例数)を先に作成しておく
        training_matrix=lil_matrix((feature_space, num_of_training_instance));
        training_data_label=[];
        instances_in_correct_label=training_map[label_name];
        for col_number, one_instance in enumerate(instances_in_correct_label):
            num_of_correct_training_instance+=1;
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
                    num_of_incorrect_training_instance+=1;
                    for feature_token in one_instance:
                        if stop==True:     
                            if feature_token not in stopwords and feature_token not in symbols:
                                feature_number=feature_map[feature_token];
                        else:
                            feature_number=feature_map[feature_token];
                        training_matrix[feature_number, col_number]=1;    
                map(lambda label: training_data_label.append(0), instances_in_incorrect_label);

        print 'Ration of correct label:incorrect label={}:{}'.format(num_of_correct_training_instance,
                                                                     num_of_incorrect_training_instance);
        tmp=training_matrix.T;
        #この場所でtrainingを実行可能な状態になっているはず
        data_train, data_test, label_train, label_test = train_test_split(tmp,\
                                                                          training_data_label);                
        #分類器にパラメータを与える
        if put_weight_constraint==True:
            print 'put weight!'
            estimator = LinearSVC(C=1.0, class_weight='auto');
            estimator.fit(data_train, label_train);
        else:
            estimator = LinearSVC(C=1.0)
            estimator.fit(data_train, label_train);
        #テストデータでチェック
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
            """

def make_format_from_training_map(token ,training_map, feature_map_character, feature_map_numeric, tfidf, one_instance_stack):
    if tfidf==True:
        if token in feature_map_character:
            for character_feature in feature_map_character[token]:
                vector_score=character_feature.split(u'_')[2];
                feature_number=feature_map_numeric[character_feature];
                one_instance_stack.append( (feature_number, vector_score) );
    elif tfidf==False:
        #liblinearのフォーマットに対応させるために(0始まりは認められないので)
        for character_feature in feature_map_character[token]:
            feature_number=feature_map_numeric[character_feature];  
            one_instance_stack.append((feature_number, 1));
    return one_instance_stack;

def split_for_train_test(out_lines_stack):
    #ここでtrainとtestに分けられるはず
    all_instances=len(out_lines_stack);
    random.shuffle(out_lines_stack);
    num_instance_for_train=int(0.95*all_instances);
    instances_for_train=[];
    for iter_index, instance in enumerate(out_lines_stack):
        instances_for_train.append(instance);
        out_lines_stack.pop(iter_index);
        if iter_index==num_instance_for_train:
            print  'breaked'
            break;
    print len(out_lines_stack)
    instances_for_test=out_lines_stack;
    return instances_for_train, instances_for_test;

def scalling_data(train_pathname):
    svmscale_exe = "/home/kensuke-mi/opt/libsvm-3.17/svm-scale";

    assert os.path.exists(train_pathname),"training file not found"
    file_name = os.path.split(train_pathname)[1]
    scaled_file = file_name + ".scale"
    model_file = file_name + ".model"
    range_file = file_name + ".range"

    cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
    print('Scaling training data...')
    #subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE).communicate()
    return scaled_file;

def out_to_libsvm_format(training_map, feature_map_numeric, feature_map_character, tfidf, exno):
    #============================================================ 
    for correct_label_key in training_map:
        ratio_map={'C':0, 'N':0};
        out_lines_stack=[];
        instances_in_correct_label=training_map[correct_label_key];
        #------------------------------------------------------------  
        #正例の処理をする
        for one_instance in instances_in_correct_label:
            ratio_map['C']+=1;
            one_instance_stack=[];
            for token in one_instance:
                one_instance_stack=make_format_from_training_map(token, 
                                                                 training_map,
                                                                 feature_map_character,
                                                                 feature_map_numeric, tfidf, one_instance_stack)
            one_instance_stack=list(set(one_instance_stack));
            one_instance_stack.sort();
            one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
            out_lines_stack.append(u'{} {}\n'.format('+1', u' '.join(one_instance_stack)));
        #------------------------------------------------------------  
        #負例の処理を行う．重みかアンダーサンプリングかのオプションを設定している
        if put_weight_constraint==True and under_sampling==False:
            for incorrect_label_key in training_map:
                if not correct_label_key==incorrect_label_key:
                    instances_in_incorrect_label=training_map[incorrect_label_key];
                    for one_instance in instances_in_incorrect_label:
                        ratio_map['N']+=1;
                        one_instance_stack=[];
                        for token in one_instance:
                            one_instance_stack=make_format_from_training_map(token, 
                                                                 training_map,
                                                                 feature_map_character,
                                                                 feature_map_numeric, tfidf, one_instance_stack)
                        one_instance_stack=list(set(one_instance_stack));
                        one_instance_stack.sort();
                        one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                        out_lines_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
                        
            ratio_c=float(ratio_map['C']) / (ratio_map['C']+ratio_map['N']);
            ratio_n=float(ratio_map['N']) / (ratio_map['C']+ratio_map['N']);
            if int(ratio_c*100)==0:
                weight_parm='-w-1 {} -w1 {} -s 2 -q'.format(1, int(ratio_n*100));
            else:
                weight_parm='-w-1 {} -w1 {} -s 2 -q'.format(int(ratio_c*100), int(ratio_n*100));
            weight_parm_svm='-w-1 {} -w1 {}'.format(int(ratio_c*100), int(ratio_n*100));
        
        #------------------------------------------------------------  
        elif put_weight_constraint==False and under_sampling==True:
            #各ラベルのインスタンス比率を求める
            num_of_incorrect_training_instance=0;
            instance_ratio_map={};
            for label in training_map:
                if label!=correct_label_key:
                    num_of_incorrect_training_instance+=len(training_map[label]);
            for label in training_map:
                if label!=correct_label_key:
                    instance_ratio_map[label]=int((float(len(training_map[label]))/num_of_incorrect_training_instance)*ratio_map['C']);
            for label in training_map:
                if label!=correct_label_key:
                    for instance_index, instances_in_incorrect_label in enumerate(training_map[label]):
                        one_instance_stack=[];
                        for token in one_instance:
                            one_instance_stack=make_format_from_training_map(token, 
                                                                             training_map,
                                                                             feature_map_character,
                                                                             feature_map_numeric, tfidf,
                                                                             one_instance_stack)
                    one_instance_stack=list(set(one_instance_stack));
                    one_instance_stack.sort();
                    one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                    out_lines_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
                               
                    if instance_index==instance_ratio_map[label]: continue;
            weight_parm='-s 2 -q';
        
        #------------------------------------------------------------  
        elif put_weight_constraint==True and under_sampling==True:
            sys.exit('Both put_weight_constraint and under_sampling is True');
        
        #------------------------------------------------------------  
        elif put_weight_constraint==False and under_sampling==False:
            for incorrect_label_key in training_map:
                if not correct_label_key==incorrect_label_key:
                    instances_in_incorrect_label=training_map[incorrect_label_key];
                    for one_instance in instances_in_incorrect_label:
                        ratio_map['N']+=1;
                        one_instance_stack=[];
                        for token in one_instance:
                            one_instance_stack=make_format_from_training_map(token, 
                                                                             training_map,
                                                                             feature_map_character,
                                                                             feature_map_numeric, tfidf,
                                                                             one_instance_stack)
                           
                    one_instance_stack=list(set(one_instance_stack));
                    one_instance_stack.sort();
                    one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                    out_lines_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
            weight_parm='';
        
        #------------------------------------------------------------  
        #ファイルに書き出しの処理をおこなう
        #インドメインでのtrainとtestに分離
        instances_for_train, instances_for_test=split_for_train_test(out_lines_stack);
        with codecs.open('./classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno, 'w', 'utf-8') as f:
            f.writelines(instances_for_train);
        with codecs.open('./classifier/libsvm_format/'+correct_label_key+'.devdata.'+exno, 'w', 'utf-8') as f:
            f.writelines(instances_for_test);
        #scalled_filepath=scalling_data('./classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno)
        train_y, train_x=svm_read_problem('./classifier/libsvm_format/'+correct_label_key+'.data.'+exno); 
        #train_y, train_x=svm_read_problem(scalled_filepath); 
        print weight_parm
        model=train(train_y, train_x, weight_parm);
        #svm_model=svm_train(train_y, train_x, weight_parm_svm);
        save_model('./classifier/liblinear/'+correct_label_key+'.liblin.model.'+exno, model);
        #svm_save_model('./classifier/libsvm/'+correct_label_key+'.svm.model', svm_model);

def main(level, mode, all_thompson_tree, stop, dutch, thompson, tfidf, exno):
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
        feature_max=1;
        num_of_training_instance=0;
        if level==1:
            construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch, thompson, tfidf, exno)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='');
    parser.add_argument('-level', '--level',
                        help='level which you want to construct big doc.', default=1)
    parser.add_argument('-mode', '--mode',
                        help='classification problem(class) or big-document(big)', required=True);
    parser.add_argument('-stop',
                        help='If added, stop words are eliminated from training file', action='store_true');
    parser.add_argument('-dutch', 
                        help='If added, document from dutch folktale database is added to training corpus', 
                        action='store_true');
    parser.add_argument('-thompson', 
                        help='If added, outline from thompson tree is added to training corpus', 
                        action='store_true');
    parser.add_argument('-tfidf', 
                        help='If added, tfidf is used for feature scoring instead of unigram feature', 
                        action='store_true');
    parser.add_argument('-exno', '--experiment_no',
                        help='save in different file',
                        default=0);
    args=parser.parse_args();
    dir_path='./parsed_json/'
    all_thompson_tree=return_range.load_all_thompson_tree(dir_path);
    result_stack=main(args.level, 
                      args.mode, 
                      all_thompson_tree, 
                      args.stop, 
                      args.dutch, 
                      args.thompson, 
                      args.tfidf,
                      args.experiment_no);
