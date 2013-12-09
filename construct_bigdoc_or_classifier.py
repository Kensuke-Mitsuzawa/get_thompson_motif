#! /usr/bin/python
# -*- coding:utf-8 -*-
"""
TODO big_document用のコード部分とclassifier部分を切り分けて，別のモジュールにしてしまうこと
いまは量が多すぎて，見難い
"""
__date__='2013/12/09'
libsvm_wrapper_path='/home/kensuke-mi/opt/libsvm-3.17/python/';
import subprocess, random, pickle, argparse, re, codecs, os, glob, json, sys;
sys.path.append(libsvm_wrapper_path);
from liblinearutil import *;
from svmutil import *;
import return_range, tf_idf, scale_grid;
import numpy;
from nltk.corpus import stopwords;
from nltk import stem;
from nltk import tokenize; 
from sklearn import svm;
from sklearn.metrics import classification_report;
from sklearn.cross_validation import train_test_split;
from sklearn.svm import LinearSVC;
from sklearn.metrics import confusion_matrix;
from scipy.sparse import lil_matrix

lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];
#option parameter
put_weight_constraint=True;
under_sampling=False;
level=1;
dev_limit=1;
#for regularization type, see README of liblinear
regularization=2;

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
    tokens=tokenize.wordpunct_tokenize(big_document_text);
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
        tokens=tokenize.wordpunct_tokenize(cleaned_sentence);
        tokens_s=[lemmatizer.lemmatize(t.lower()) for t in tokens]
        if stop==True:
            tokens_set_stack.append([t for t in tokens_s if t not in stopwords and t not in symbols]);
        else:
            tokens_set_stack.append(tokens_s) 
    return tokens_set_stack;

def make_feature_set(feature_map, label_name, tokens_set_stack, feature_mode, stop, args):
    """
    素性関数を作り出す（要はただのmap）
    12/07 easy domain adaptation用に書き換え．(３種の素性：t:thompson, d:dutch, g:general)
    12/09 easy domain modeとそうでないmodeに切り分け
    feature_modeがdutchの時：頭文字にd,と頭文字にgの素性作成
    feature_modeがthompsonの時：頭文字にt,と頭文字にgの素性作成
    """
    easy_domain=args.easy_domain;
    if easy_domain==True:
        #各ドメインの識別用頭文字を設定
        if feature_mode=='thompson':
            prefix=u't';
        elif feature_mode=='dutch':
            prefix=u'd';
    elif easy_domain==False:
        prefix='normal';

    for token_instance in tokens_set_stack:
        for token in token_instance:
            #ドメインごとの素性を登録
            hard_cluster_feature=u'{}_{}_unigram'.format(prefix, token);
            character_feature=hard_cluster_feature;
            if stop==True and token not in stopwords and token not in symbols:
                if token in feature_map and character_feature not in feature_map[token]:
                    feature_map[token].append(character_feature);
                elif token in feature_map and character_feature in feature_map[token]:
                    pass; 
                else:
                    feature_map[token]=[character_feature];
            elif stop==False:
                if token in feature_map and character_feature not in feature_map[token]:
                    feature_map[token].append(character_feature);
                elif token in feature_map and character_feature in feature_map[token]:
                    pass; 
                else:
                    feature_map[token]=[character_feature];
            
            if easy_domain==True:
                #general用の素性を登録
                #generalを登録するのはeasy domainがTrueの時だけ
                general_feature=u'{}_{}_unigram'.format('g', token);
                character_feature=general_feature;
                if stop==True and token not in stopwords and token not in symbols:
                    if token in feature_map and character_feature not in feature_map[token]:
                        feature_map[token].append(character_feature);
                    elif token in feature_map and character_feature in feature_map[token]:
                        pass; 
                    else:
                        feature_map[token]=[character_feature];
                elif stop==False:
                    if token in feature_map and character_feature not in feature_map[token]:
                        feature_map[token].append(character_feature);
                    elif token in feature_map and character_feature in feature_map[token]:
                        pass;
                    else:
                        feature_map[token]=[character_feature];
    #ここはsoftラベル用の素性を見ていた時のコード
    """
    #allはすでにsoftの素性が存在しているときに付与
    #elif feature_mode=='all':
    elif feature_mode=='dutch':
        for token_instance in tokens_set_stack:
            for token in token_instance:
                normal_cluster_feature=u'{}_{}_unigram'.format(prefix, token);
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
    """
    return feature_map;

def make_tfidf_feature_from_score(tfidf_score_map, wordset_map, feature_map_character, args):
    """
    TFIDF用の素性を作成する．
    easy_domainを適用するときと，そうでないときで区別
    """
    if args.easy_domain==True:
        for token_key in tfidf_score_map:
            general_format=u'g_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key]);
            if token_key in wordset_map['thompson']:
                thompson_format=u't_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key]); 
                if token_key not in feature_map_character: 
                    feature_map_character[token_key]=[thompson_format];
                if token_key in feature_map_character and general_format not in feature_map_character[token_key]:
                    feature_map_character[token_key].append(u'g_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key])); 
                if token_key in feature_map_character and thompson_format not in feature_map_character[token_key]:
                    feature_map_character[token_key].append(thompson_format);
            #この下のifを上と同じように書き換え
            if token_key in wordset_map['dutch']:
                dutch_format=u'd_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key]); 
                if token_key not in feature_map_character: 
                    feature_map_character[token_key]=[thompson_format];
                if token_key in feature_map_character and general_format not in feature_map_character[token_key]:
                    feature_map_character[token_key].append(u'g_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key])); 
                if token_key in feature_map_character and dutch_format not in feature_map_character[token_key]:
                    feature_map_character[token_key].append(dutch_format);
            #ここ，不安なんだけど．．テストのコーパスはターゲットドメインのはずだから，ターゲットを表すdでいいはず．．．
            if token_key in wordset_map['test']:
                test_format=u'd_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key]); 
                if token_key not in feature_map_character: 
                    feature_map_character[token_key]=[thompson_format];
                if token_key in feature_map_character and general_format not in feature_map_character[token_key]:
                    feature_map_character[token_key].append(u'g_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key])); 
                if token_key in feature_map_character and test_format not in feature_map_character[token_key]:
                    feature_map_character[token_key].append(test_format);
    elif args.easy_domain==False:
        #easy domain adaptationを使わないモード用にここを残しておく 
        for token_key in tfidf_score_map:
            normal_format=u'normal_{}_{}_tfidf'.format(token_key, tfidf_score_map[token_key]);
            if token_key not in feature_map_character:
                feature_map_character[token_key]=[normal_format];
            elif token_key in feature_map_character:
                feature_map_character[token_key].append(normal_format);
    return feature_map_character;

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

def construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch, thompson, tfidf, exno, args):
    dev_mode=args.dev;
    exno=str(exno);
    training_map={};
    tfidf_score_map={};
    feature_map_character={};
    num_of_training_instance=0;
    #============================================================ 
    #オランダ語コーパスとトンプソン木をフラッグによって，訓練に使うかどうかを分岐
    if dutch==True:
        dutch_training_map={};
        #古い方のパス（自分で翻訳してた頃）
        #dir_path='../dutch_folktale_corpus/given_script/translated_big_document/leaf_layer/' 
        #新しい方のパス(translated by kevin's system)
        #dir_path='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated/'
        #訓練用に分割したディレクトリ
        dir_path='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated_train/'
        #description付きのバージョンなら
        #dir_path='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated/'
        #------------------------------------------------------------
        #文書を全部よみこんで，training_mapの下に登録する．前処理みたいなもん
        for fileindex, filepath in enumerate(make_filelist(dir_path)):
            if level==1:
                alphabet_label_list=(os.path.basename(filepath)).split('_')[:-1];
            elif level==2:
                alphabet_label=(os.path.basename(filepath))[0];
            tokens_in_label=tokenize.wordpunct_tokenize(codecs.open(filepath, 'r', 'utf-8').read());
            lemmatized_tokens_in_label=[lemmatizer.lemmatize(t.lower()) for t in tokens_in_label];
            if stop==True:
                lemmatized_tokens_in_label=[t for t in lemmatized_tokens_in_label if t not in stopwords and t not in symbols];
            if level==1:
                for alphabet_label in alphabet_label_list:
                    alphabet_label=alphabet_label.upper();
                    if alphabet_label in dutch_training_map:
                        dutch_training_map[alphabet_label].append(lemmatized_tokens_in_label);
                    else:
                        dutch_training_map[alphabet_label]=[lemmatized_tokens_in_label];
            elif level==2:
                alphabet_label=alphabet_label.upper();
                if alphabet_label in training_map:
                    dutch_training_map[alphabet_label].append(lemmatized_tokens_in_label);
                else:
                    dutch_training_map[alphabet_label]=[lemmatized_tokens_in_label];
            if dev_mode==True and fileindex==dev_limit:
                break;
        #最後にtraining_mapの下に登録
        training_map['dutch']=dutch_training_map;
        #------------------------------------------------------------ 
        #training_mapへの登録が全部おわってから，素性抽出を行う 
        #easy domain adaptation用にここで工夫ができるはず
        if tfidf==False:
            #A~Zのラベル間でcapな単語を求めだす
            #全ラベル間でcapな単語を作成して，{token}:'capなラベルをアンダースコア接続で表現'
            #TODO この関数にミスがあると思う．複数のラベルが取得できていない
            #feature_map_character=make_soft_char_feature(dutch_training_map, feature_map_character, stop);
            doc_token=[];
            for label in dutch_training_map:
                doc=dutch_training_map[label];
                doc_token+=doc; 
            feature_map_character=make_feature_set(feature_map_character, None, doc_token, 'dutch', stop, args);
        #------------------------------------------------------------ 
            #tfidf用のコードがあった跡地
        for alphabet_label in dutch_training_map:
            print u'The num. of training instance for {} in dutch corpus is {}'.format(alphabet_label, len(dutch_training_map[alphabet_label]));
        print u'-'*30;
    #============================================================ 
    #Thompsonのインデックスツリーを訓練データに加える
    if thompson==True:
        thompson_training_map={};
        for key_index, key_1st in enumerate(all_thompson_tree):
            parent_node=key_1st;
            class_training_stack=construct_class_training_1st(parent_node, all_thompson_tree);
            tokens_set_stack=cleanup_class_stack(class_training_stack, stop);
            
            print u'-'*30;
            print u'Training instances for {} from thompson tree:{}'.format(key_1st,len(tokens_set_stack));
            num_of_training_instance+=len(tokens_set_stack);
            #------------------------------------------------------------ 
            #作成した文書ごとのtokenをtrainingファイルを管理するmapに追加
            #TFIDFがTrueだろうが，Falseだろうが関係なく，ここは実行される
            if key_1st in thompson_training_map:
                thompson_training_map[key_1st]+=tokens_set_stack;
            else:
                thompson_training_map[key_1st]=tokens_set_stack;
            if dev_mode==True and key_index==dev_limit:
                break;
        #------------------------------------------------------------ 
        #素性をunigram素性にする
        if tfidf==False:
            for label in thompson_training_map:
                tokens_set_stack=thompson_training_map[label];
                #文字情報の素性関数を作成する
                feature_map_character=make_feature_set(feature_map_character,
                                                       label, tokens_set_stack, 'thompson', stop, args);
        #------------------------------------------------------------ 
            #tdidf用のコードがあった跡地
        training_map['thompson']=thompson_training_map;
    #============================================================ 
    #もしTFIDFを使うのであれば，test documentも合わせた空間で重みスコアを求めないといけない
    if tfidf==True:
        #------------------------------------------------------------
        #TFIDFスコアはthompson resourceとdutch_folktale_corpusとtest documentのすべてを合わせた空間で求めないといけない
        all_training_instances=[];
        #静的にハードコーディングはあまりしたくないんだけど．．仕方がない
        wordset_map={'thompson':[], 'dutch':[], 'test':[]};
        for subdata in training_map:
            training_instances=[];
            for label in training_map[subdata]: 
                tokens_in_docs_in_label=training_map[subdata][label];
                #マルチラベルのために，training_mapの中には重複して同じ文書が登録されていることがある．
                #重複した文書が追加されないための措置．
                #実際には，空の文書がいくつかあるので，training_instaces_dutchの要素数はファイル数よりは減るはず
                for tokens_in_doc in tokens_in_docs_in_label:
                    if tokens_in_doc not in training_instances:
                        all_training_instances.append(tokens_in_doc);
                        training_instances.append(tokens_in_doc);
            #ちょっと変な書き方だけど，この方が早い
            wordset_map[subdata]=[t for doc in training_instances for t in doc];
        #------------------------------------------------------------
        #ペルシア語口承文芸コーパスからファイルを読み込む
        test_corpus_instances=[];
        persian_folktale_documet_path='../corpus_dir/translated_google/'
        for doc_filepath in make_filelist(persian_folktale_documet_path):
            doc=[];
            with codecs.open(doc_filepath, 'r', 'utf-8') as lines:
                for line in lines:
                    if line==u'#motif\n':
                        motif_line_flag=True; 
                        continue;
                    elif line==u'#text\n':
                        motif_line_flag=False;
                        text_line_flag=True;
                        continue;
                    elif motif_line_flag==True:
                        pass;
                    elif text_line_flag==True:
                        doc.append(line);
            doc=u' '.join(doc); 
            tokens=tokenize.wordpunct_tokenize(doc);
            lemmatized_tokens=[lemmatizer.lemmatize(t.lower()) for t in tokens];
            if stop==True:
                lemmatized_tokens=[t for t in lemmatized_tokens if t not in stopwords and t not in symbols];
            test_corpus_instances.append(lemmatized_tokens);
            wordset_map['test'].append([t for t in lemmatized_tokens]);
        #ちょっと変な書き方だけど，この方が実行速度はやい
        wordset_map['test']=[t for doc in wordset_map['test'] for t in doc];
        training_plus_test_docs=all_training_instances+test_corpus_instances;
        #------------------------------------------------------------
        print 'TFIDF score calculating'
        tfidf_score_map=tf_idf.tf_idf_test(training_plus_test_docs);
        feature_map_character=make_tfidf_feature_from_score(tfidf_score_map,
                                                            wordset_map,
                                                            feature_map_character, args);
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
    
    if args.training=='liblinear':
        #liblinearを使ったモデル作成
        out_to_libsvm_format(training_map, 
                            feature_map_numeric, 
                            feature_map_character,
                            tfidf,
                            tfidf_score_map,
                            exno, args);
    elif args.training=='mulan':
        dutch_dir_path='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated_train/'
        #mulanを使ったモデル作成
        #training_mapは使えないので新たにデータ構造の再構築をする（もったいないけど）
        #thompson木は元々マルチラベルでもなんでもないので，使わない
        training_data_list=create_multilabel_datastructure(dutch_dir_path, args); 
        #TODO 新しくlabel spaceを定義したまだ未定義なのではやく処理する
        #欲しいのは，ラベルアルファベットの配列 [A, B, C, D, E....]
        out_to_mulan_format(training_data_list, 
                            feature_map_numeric, 
                            feature_map_character,
                            tfidf,
                            tfidf_score_map,
                            exno, feature_space, 
                            label_space, args);


def training_with_scikit():
    pass;
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

def convert_to_feature_space(training_map,
                            feature_map_character,
                            feature_map_numeric,
                             tfidf_score_map, tfidf, args):
    if args.training=='liblinear':
        """
        training_mapの中身を素性空間に変換する．
        戻り値は数値表現になった素性空間．
        データ構造はtraining_mapと同じ．（ただし，最後だけtokenでなく，素性番号：素性値のタプル）
        """
        training_map_feature_space={};
        for subdata in training_map:
            if args.easy_domain==True:
                if subdata=='dutch':
                    prefix=u'd';
                elif subdata=='thompson':
                    prefix=u't';
            elif args.easy_domain==False:
                prefix=u'normal';
            feature_space_label={};
            #------------------------------------------------------------     
            for label in training_map[subdata]:
                #------------------------------------------------------------     
                #token表現を文書ごとに素性空間にマップする．
                for doc in training_map[subdata][label]:
                    feature_space_doc=[];
                    for token in doc:
                        #feature_map_character[token]の中は配列になっているので，資源にあった資源のみを選ぶ
                        if token in feature_map_character: 
                            for candidate in feature_map_character[token]:
                                feature_pattern=re.compile(u'^{}'.format(prefix));
                                if re.search(feature_pattern, candidate):
                                    domain_feature=candidate;
                                    domain_feature_numeric=feature_map_numeric[domain_feature];
                                    if tfidf==False:
                                        feature_space_doc.append((domain_feature_numeric,
                                                                  1)); 
                                    #ここがtfidfが真の場合は，素性値をタプルにして追加すればよい
                                    elif tfidf==True:
                                        feature_space_doc.append((domain_feature_numeric,
                                                                  tfidf_score_map[token]));
                                if re.search(ur'^g', candidate):
                                    general_feature=candidate;
                                    general_feature_numeric=feature_map_numeric[general_feature];
                                    if tfidf==False:
                                        feature_space_doc.append((general_feature_numeric,
                                                                  1));
                                    #ここがtfidfが真の場合は，素性値をタプルにして追加すればよい
                                    elif tfidf==True:
                                        feature_space_doc.append((general_feature_numeric,
                                                                  tfidf_score_map[token]));
                    #------------------------------------------------------------     
                    if label not in feature_space_label:
                        feature_space_label[label]=[feature_space_doc];
                    else:
                        feature_space_label[label].append(feature_space_doc);
                #------------------------------------------------------------     
            training_map_feature_space[subdata]=feature_space_label;
            #------------------------------------------------------------     
        return training_map_feature_space;
    #============================================================ 
    elif args.training=='mulan':
        training_data_list_feature_space=[];
        training_data_list=training_map;
        for one_instance in training_data_list:
            one_instance_stack=[];
            for token in one_instance[1]:
                if token in feature_map_character:
                    for feature_candidate in feature_map_character[token]:
                        feature_number=feature_map_numeric[feature_candidate];
                        one_instance_stack.append(feature_number);
            training_data_list_feature_space.append((one_instance[0], one_instance_stack));
        return training_data_list_feature_space;

def unify_tarining_feature_space(training_map_feature_space):
    unified_map={};
    for subdata_key in training_map_feature_space:
        for label in training_map_feature_space[subdata_key]:
            if label not in unified_map:
                unified_map[label]=training_map_feature_space[subdata_key][label];
            else:
                unified_map[label]+=training_map_feature_space[subdata_key][label];
    return unified_map;

def make_format_from_training_map(token ,training_map, feature_map_character, feature_map_numeric, tfidf, one_instance_stack):
    #この関数は不要になったのでそのうち削除
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

def split_for_train_test(correct_instances_stack, incorrect_instances_stack, instace_lines_num_map, ratio_of_training_instance):
    #ここでtrainとtestに分けられるはず
    all_instances=len(correct_instances_stack)+len(incorrect_instances_stack);
    random.shuffle(correct_instances_stack);
    random.shuffle(incorrect_instances_stack);
    ratio_of_correct=float(instace_lines_num_map['C'])/(instace_lines_num_map['C']+instace_lines_num_map['N']);
    ratio_of_incorrect=float(instace_lines_num_map['N'])/(instace_lines_num_map['C']+instace_lines_num_map['N']);
    #訓練用のインスタンス数
    #headerの変数で量を調整可能
    num_instance_for_train=int(ratio_of_training_instance*all_instances);
    #テスト用のインスタンス数
    num_instance_for_test=all_instances-num_instance_for_train;
    #正例と負例の事例スタックから何行ずつとって来ればいいのか？を計算
    num_of_instances_of_correct_for_test=int(num_instance_for_test*ratio_of_correct);
    num_of_instances_of_incorrect_for_test=int(num_instance_for_test*ratio_of_incorrect);
    #スライス機能を使って，テスト用のインスタンスを獲得
    instances_for_test=correct_instances_stack[:num_of_instances_of_correct_for_test]\
            +incorrect_instances_stack[:num_of_instances_of_incorrect_for_test];
    #スライス機能を使って，訓練用のインスタンスを獲得
    instances_for_train=correct_instances_stack[num_of_instances_of_correct_for_test:]\
            +incorrect_instances_stack[num_of_instances_of_incorrect_for_test:];
    return instances_for_train, instances_for_test;

def close_test(classifier_path, test_path):
    print 'close test result for {} with {}'.format(classifier_path, test_path);
    y, x = svm_read_problem(test_path);
    m = load_model(classifier_path);
    p_label, p_acc, p_val = predict(y, x, m);
    ACC, MSE, SCC = evaluations(y, p_label); 
    print ACC, MSE, SCC;

def create_multilabel_datastructure(dir_path, args):
    training_data_list=[];
    level=args.level;
    for fileindex, filepath in enumerate(make_filelist(dir_path)):
        if level==1:
            alphabet_label_list=(os.path.basename(filepath)).split('_')[:-1];
        elif level==2:
            alphabet_label=(os.path.basename(filepath))[0];
        tokens_in_label=tokenize.wordpunct_tokenize(codecs.open(filepath, 'r', 'utf-8').read());
        lemmatized_tokens_in_label=[lemmatizer.lemmatize(t.lower()) for t in tokens_in_label];
        if args.stop==True:
            lemmatized_tokens_in_label=\
                    [t for t in lemmatized_tokens_in_label if t not in stopwords and t not in symbols];
        if level==1:
            #ラベル列，token列のタプルにして追加
            training_data_list.append(([alphabet_label.upper() for alphabet_label in alphabet_label_list],
                                      lemmatized_tokens_in_label));
        #level2のことは後で考慮すればよい
        """
        elif level==2:
            alphabet_label=alphabet_label.upper();
            if alphabet_label in training_map:
                dutch_training_map[alphabet_label].append(lemmatized_tokens_in_label);
            else:
                dutch_training_map[alphabet_label]=[lemmatized_tokens_in_label];
        if dev_mode==True and fileindex==dev_limit:
            break;
            """
    return training_data_list;
def out_to_mulan_format(training_data_list, feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        exno, feature_space, label_space, args):
    training_data_list_feature_space=convert_to_feature_space(training_data_list,
                                                            feature_map_character,
                                                            feature_map_numeric,
                                                            tfidf_score_map, tfidf, args);
    mulan_header_relation=u'@relation {}\n\n';
    #TODO headerに素性の名前を記入していかないといけない!
    #------------------------------------------------------------
    for one_instance in training_data_list_feature_space:
        feature_space_for_one_instance=[0]*feature_space;
        for feature_number_tuple in one_instance[1]:
            #今はunigramの想定で書いてるけど，後でtfidf用も書き加えないといけない
            #だから，変数名はtupleになっている（実際にはint型が入ってる）
            feature_space_for_one_instance[feature_number_tuple]=1;
        

def out_to_libsvm_format(training_map_original, feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        exno, args):
    training_map_feature_space=convert_to_feature_space(training_map_original,
                                                        feature_map_character,
                                                        feature_map_numeric,
                                                        tfidf_score_map, tfidf, args);
    unified_training_map=unify_tarining_feature_space(training_map_feature_space); 
    training_map=unified_training_map; 
    #============================================================ 
    for correct_label_key in training_map:
        instance_lines_num_map={'C':0, 'N':0};
        lines_for_correct_instances_stack=[];
        lines_for_incorrect_instances_stack=[];
        out_lines_stack=[];
        instances_in_correct_label=training_map[correct_label_key];
        #------------------------------------------------------------  
        #正例の処理をする
        for one_instance in instances_in_correct_label:
            instance_lines_num_map['C']+=1;
            one_instance_stack=one_instance;
            one_instance_stack=list(set(one_instance_stack));
            one_instance_stack.sort();
            one_instance=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
            lines_for_correct_instances_stack.append(u'{} {}\n'.format('+1', u' '.join(one_instance)));
        #------------------------------------------------------------  
        #負例の処理を行う．重みかアンダーサンプリングかのオプションを設定している
        if put_weight_constraint==True and under_sampling==False:
            for incorrect_label_key in training_map:
                if not correct_label_key==incorrect_label_key:
                    instances_in_incorrect_label=training_map[incorrect_label_key];
                    for one_instance in instances_in_incorrect_label:
                        #仮にこの変数名にしておく
                        one_instance_stack=one_instance; 
                        instance_lines_num_map['N']+=1;
                        one_instance_stack=list(set(one_instance_stack));
                        one_instance_stack.sort();
                        one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                        lines_for_incorrect_instances_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
            ratio_c=float(instance_lines_num_map['C']) / (instance_lines_num_map['C']+instance_lines_num_map['N']);
            ratio_n=float(instance_lines_num_map['N']) / (instance_lines_num_map['C']+instance_lines_num_map['N']);
            if int(ratio_c*100)==0:
                weight_parm='-w-1 {} -w1 {} -s {} -q'.format(1, int(ratio_n*100), regularization);
            else:
                weight_parm='-w-1 {} -w1 {} -s {} -q'.format(int(ratio_c*100), int(ratio_n*100), regularization);
            weight_parm_svm='-w-1 {} -w1 {}'.format(int(ratio_c*100), int(ratio_n*100));
        #------------------------------------------------------------  
        elif put_weight_constraint==False and under_sampling==True:
            #各ラベルのインスタンス比率を求める
            num_of_incorrect_training_instance=0;
            instance_ratio_map={};
            #負例の数を計算
            for label in training_map:
                if label!=correct_label_key:
                    num_of_incorrect_training_instance+=len(training_map[label]);
                    instance_lines_num_map['N']+=len(training_map[label]);
            #負例のうちの特定のラベルが何行分出力すれば良いのか？を計算する
            for label in training_map:
                if label!=correct_label_key:
                    instance_ratio_map[label]=\
                            int((float(len(training_map[label]))/num_of_incorrect_training_instance)*instance_lines_num_map['C']);
            for label in training_map:
                if label!=correct_label_key:
                    for instance_index, one_instance in enumerate(training_map[label]):
                        #あとで変数名を変えておくこと，これだと意味が違う
                        one_instance_stack=one_instance;
                        one_instance_stack=list(set(one_instance_stack));
                        one_instance_stack.sort();
                        one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                        lines_for_incorrect_instances_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
                    #比率にもとづいて計算された行数を追加し終わったら，次のラベルに移る 
                    if instance_index==instance_ratio_map[label]: continue;
            weight_parm='-s {} -q'.format(regularization);
        #------------------------------------------------------------  
        elif put_weight_constraint==True and under_sampling==True:
            sys.exit('[Warning] Both put_weight_constraint and under_sampling is True');
        
        #------------------------------------------------------------  
        elif put_weight_constraint==False and under_sampling==False:
            for incorrect_label_key in training_map:
                if not correct_label_key==incorrect_label_key:
                    instances_in_incorrect_label=training_map[incorrect_label_key];
                    for one_instance in instances_in_incorrect_label:
                        instance_lines_num_map['N']+=1;
                        one_instance_stack=one_instance;
                        one_instance_stack=list(set(one_instance_stack));
                        one_instance_stack.sort();
                        one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                        lines_for_incorrect_instances_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
            weight_parm='-s {}'.format(regularization);
        
        #ここでfeature_spaceに変換されたmapがあると良い．
        #で，mixedしてから次の処理に渡す
        #------------------------------------------------------------  
        #ファイルに書き出しの処理をおこなう
        #インドメインでのtrainとtestに分離
        training_amount=float(args.training_amount);
        print u'Training : Test ratio={} : {}'.format(training_amount, 1-training_amount);
        instances_for_train, instances_for_test=split_for_train_test(lines_for_correct_instances_stack,
                                                                     lines_for_incorrect_instances_stack,
                                                                     instance_lines_num_map,
                                                                     training_amount);
        with codecs.open('./classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno, 'w', 'utf-8') as f:
            f.writelines(instances_for_train);
        with codecs.open('./classifier/libsvm_format/'+correct_label_key+'.devdata.'+exno, 'w', 'utf-8') as f:
            f.writelines(instances_for_test);
        return_value=scale_grid.main('./classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno,
                        './classifier/libsvm_format/'+correct_label_key+'.devdata.'+exno,
                        False);
        weight_parm+=u' -c {} -p {}'.format(return_value[0], return_value[1]);
        train_y, train_x=svm_read_problem('./classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno); 
        print weight_parm
        model=train(train_y, train_x, weight_parm);
        save_model('./classifier/liblinear/'+correct_label_key+'.liblin.model.'+exno, model);
        close_test('./classifier/liblinear/'+correct_label_key+'.liblin.model.'+exno,
                   './classifier/libsvm_format/'+correct_label_key+'.devdata.'+exno);
        os.remove('{}.traindata.{}.out'.format(correct_label_key, exno));
        print u'-'*30;
        #train_y, train_x=svm_read_problem(scalled_filepath); 
        #svm_model=svm_train(train_y, train_x, weight_parm_svm);
        #svm_save_model('./classifier/libsvm/'+correct_label_key+'.svm.model', svm_model);

def main(level, mode, all_thompson_tree, stop, dutch, thompson, tfidf, exno, args):
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
            construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch, thompson, tfidf, exno, args)

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
    parser.add_argument('-dev', '--dev',
                        help='developping mode',
                        action='store_true');
    parser.add_argument('-training_amount', '--training_amount',
                        help='The ratio of training amount',
                        default=0.95);
    parser.add_argument('-easy_domain', '--easy_domain',
                        help='use easy domain adaptation',
                        action='store_true');
    parser.add_argument('-training', help='which training tool?',
                        required=True);
    args=parser.parse_args();
    dir_path='./parsed_json/'
    #------------------------------------------------------------    
    if float(args.training_amount)>=1.0:
        sys.exit('[Warning] -training_amount must be between 0-1(Not including 1)');
    #------------------------------------------------------------    
    if args.easy_domain==True:
        if not (args.dutch==True and args.thompson==True):
            sys.exit('[Warning] You specified easy_domain mode. But there is only one domain');
    #------------------------------------------------------------    
    if not args.training=='liblinear' and not args.training=='mulan':
        sys.exit('[Warning] choose correct training tool');
    #------------------------------------------------------------    
    all_thompson_tree=return_range.load_all_thompson_tree(dir_path);
    result_stack=main(args.level, 
                      args.mode, 
                      all_thompson_tree, 
                      args.stop, 
                      args.dutch, 
                      args.thompson, 
                      args.tfidf,
                      args.experiment_no,
                      args);
