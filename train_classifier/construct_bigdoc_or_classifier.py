#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2013/12/21';

import argparse, re, codecs, os, glob, json, sys;
sys.path.append('../');
import return_range, tf_idf, mulan_module, liblinear_module, bigdoc_module;
from nltk.corpus import stopwords;
from nltk import stem;
from nltk import tokenize; 

lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];
#option parameter
level=1;
dev_limit=1;
#Idea number of TFIDF
tfidf_idea=2;

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

def preprocess_particular_case(sentence):
    if re.search(ur'\w\.\w', sentence):
        sentence=re.sub(ur'(\w)\.(\w)', ur'\1 \2', sentence);
        return sentence; 
    else:
        return sentence;

def cleanup_class_stack(class_training_stack, stop):
    tokens_set_stack=[];
    for tuple_item in class_training_stack:
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

def create_tfidf_feat_idea1(training_map, feature_map_character, args):
    stop=args.stop;
    #------------------------------------------------------------
    #TFIDFスコアはthompson resourceとdutch_folktale_corpusとtest documentのすべてを合わせた空間で求める
    #TODO L2正則化できるようにすること
    #TODO ゴミみたいな重みの語はすててもよいのでは？
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
    persian_folktale_documet_path='../../corpus_dir/translated_google/'
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
    print 'TFIDF(Idea-1) score calculating'
    tfidf_score_map=tf_idf.tf_idf_test(training_plus_test_docs);
    feature_map_character=make_tfidf_feature_from_score(tfidf_score_map,
                                                        wordset_map,
                                                        feature_map_character, args);
    return feature_map_character, tfidf_score_map;
    
def create_tfidf_feat_idea2(training_map, feature_map_character, args):
    #ラベルごとの重要語を取り出す
    #TFIDFスコアを文書集合から算出した後，ラベル文書ごとに閾値（足切り値）を求め，閾値以下の語は素性を作らない
    #これで，「あるラベルに特徴的な語」を示す素性が作れた．と思う
    import math;
    bigdocs_stack=[];
    bigdocs_stack_nolabel=[];
    #------------------------------------------------------------
    #２つの資源から混合の文書集合を作成
    for label in training_map['thompson']:
        instances_in_label_thompson=training_map['thompson'][label];
        bigdoc_of_label_thompson=[t for doc in instances_in_label_thompson for t in doc];        
        if label in training_map['dutch']:        
            instances_in_label_dutch=training_map['dutch'][label];        
            bigdoc_of_label_dutch=[t for doc in instances_in_label_dutch for t in doc];        
            bigdoc_of_label_thompson+=bigdoc_of_label_dutch;
        bigdocs_stack.append((label, bigdoc_of_label_thompson));
        bigdocs_stack_nolabel.append(bigdoc_of_label_thompson);
    #------------------------------------------------------------
    print 'TFIDF(Idea-2) score calculating'
    tfidf_score_map=tf_idf.tf_idf_test(bigdocs_stack_nolabel);     
    #------------------------------------------------------------
    #L2正則化をかける        
    weight_sum=0;    
    for key in tfidf_score_map:
        weight=tfidf_score_map[key];
        weight_sum+=(weight)**2;
    L2_norm=math.sqrt(weight_sum);
    L2_normalized_map={};    
    for key in tfidf_score_map:
        normalized_score=tfidf_score_map[key]/L2_norm;
        L2_normalized_map[key]=normalized_score;        
    
    for one_doc_tuple in bigdocs_stack:
        label=one_doc_tuple[0];
        tokens=one_doc_tuple[1];
        weightsum_of_label=0;
        for t in tokens:
            if t in L2_normalized_map:
                word_weight=L2_normalized_map[t];
                weightsum_of_label+=word_weight;
        #閾値を，重み平均に設定する(中央値でも良い気がする)
        threshold_point=weightsum_of_label/len(L2_normalized_map);        
        for t in tokens:            
            if t in L2_normalized_map:
                #対数化しているので，符号が逆になる
                if L2_normalized_map[t] > threshold_point:  
                    pass;
                else:
                    weight_format=u'{}_{}_{}'.format(label, t, word_weight);
                    if t not in feature_map_character:
                        feature_map_character[t]=[weight_format];
                    elif weight_format not in feature_map_character[t]:
                        feature_map_character[t].append(weight_format);
                            
    return feature_map_character, L2_normalized_map;

def construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch, thompson, tfidf, args):
    dev_mode=args.dev;
    exno=str(args.experiment_no);
    motif_vector=[unichr(i) for i in xrange(65,65+26)];
    motif_vector.remove(u'O'); motif_vector.remove(u'I');
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
        dir_path='../../dutch_folktale_corpus/dutch_folktale_database_translated_kevin_system/translated_train/'
        #クエリ翻訳の方式
        #dir_path='../dutch_folktale_corpus/dutch_folktale_database_query_translated_google_translated_train/';
        #description付きのバージョンなら
        #dir_path='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated/'
        #------------------------------------------------------------
        #文書を全部よみこんで，training_mapの下に登録する．前処理みたいなもん
        for fileindex, filepath in enumerate(make_filelist(dir_path)):
            if level==1:
                alphabet_label_list=(os.path.basename(filepath)).split('_')[:-1];
            elif level==2:
                alphabet_label=(os.path.basename(filepath))[0];
            file_obj=codecs.open(filepath, 'r', 'utf-8');
            tokens_in_label=tokenize.wordpunct_tokenize(file_obj.read());
            file_obj.close(); 
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
        training_map['thompson']=thompson_training_map;
    #============================================================ 
    #もしTFIDFを使うのであれば，test documentも合わせた空間で重みスコアを求めないといけない
    if tfidf==True:
        if tfidf_idea==1:
            feature_map_character, tfidf_score_map=create_tfidf_feat_idea1(training_map, feature_map_character, args);
        elif tfidf_idea==2:
            feature_map_character, tfidf_score_map=create_tfidf_feat_idea2(training_map, feature_map_character, args);
    #============================================================  
    #作成した素性辞書をjsonに出力(TFIDF)が空の時は空の辞書が出力される
    with codecs.open('../classifier/tfidf_weight/tfidf_word_weight.json.'+exno, 'w', 'utf-8') as f:
        json.dump(tfidf_score_map, f, indent=4, ensure_ascii=False);
    with codecs.open('../classifier/feature_map_character/feature_map_character_1st.json.'+exno, 'w', 'utf-8') as f:
        json.dump(feature_map_character, f, indent=4, ensure_ascii=False);
    #ここで文字情報の素性関数を数字情報の素性関数に変換する
    feature_map_numeric=make_numerical_feature(feature_map_character);
    
    with codecs.open('../classifier/feature_map_numeric/feature_map_numeric_1st.json.'+exno, 'w', 'utf-8') as f:
        json.dump(feature_map_numeric, f, indent=4, ensure_ascii=False);

    feature_space=len(feature_map_numeric);
    print u'The number of feature is {}'.format(feature_space)
    
    if args.training=='liblinear':
        #liblinearを使ったモデル作成
        liblinear_module.out_to_libsvm_format(training_map, 
                            feature_map_numeric, 
                            feature_map_character,
                            tfidf,
                            tfidf_score_map,
                            exno, args);
    elif args.training=='mulan':
        dutch_dir_path='../../dutch_folktale_corpus/dutch_folktale_database_translated_kevin_system/translated_train/'
        #mulanを使ったモデル作成
        #training_mapは使えないので新たにデータ構造の再構築をする（もったいないけど）
        #thompson木は元々マルチラベルでもなんでもないので，使わない
        training_data_list=create_multilabel_datastructure(dutch_dir_path, args); 
        if args.thompson==True:
            training_data_list=create_multilabel_datastructure_single(training_data_list,
                                                   thompson_training_map,
                                                   args);
        mulan_module.out_to_mulan_format(training_data_list, 
                            feature_map_numeric, 
                            feature_map_character,
                            tfidf, tfidf_score_map,
                            feature_space, 
                            motif_vector, args);

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

                                if re.search(ur'^[A-Z]_', candidate):
                                    domain_feature=candidate;
                                    domain_feature_numeric=feature_map_numeric[domain_feature];
                                    if tfidf==False:
                                        feature_space_doc.append((domain_feature_numeric,
                                                                  1)); 
                                    #ここがtfidfが真の場合は，素性値をタプルにして追加すればよい
                                    elif tfidf==True:
                                        if token in tfidf_score_map:
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
                    if not feature_space_doc==[]:
                        if label not in feature_space_label:
                            feature_space_label[label]=[feature_space_doc];
                        else:
                            feature_space_label[label].append(feature_space_doc);
                #------------------------------------------------------------
            if not feature_space_label=={}:
                training_map_feature_space[subdata]=feature_space_label;
            #------------------------------------------------------------    
        return training_map_feature_space;
    #============================================================ 
    elif args.training=='mulan':
        """
        RETURN list [tuple (list [unicode モチーフラベル], list [unicode token])] 
        """
        training_data_list_feature_space=[];
        training_data_list=training_map;
        for one_instance in training_data_list:
            one_instance_stack=[];
            for token in one_instance[1]:
                if args.tfidf==False:
                    if token in feature_map_character:
                        for feature_candidate in feature_map_character[token]:
                            feature_number=feature_map_numeric[feature_candidate];
                            one_instance_stack.append((feature_number, 1));
                elif args.tfidf==True:
                    if token in tfidf_score_map:
                        for feature_candidate in feature_map_character[token]:
                            feature_number=feature_map_numeric[feature_candidate];
                            tfidf_weight=tfidf_score_map[token];
                            one_instance_stack.append((feature_number, tfidf_weight));

            training_data_list_feature_space.append((one_instance[0], one_instance_stack));
        return training_data_list_feature_space;

def create_multilabel_datastructure_single(training_data_list, thompson_training_map, args):
    """
    mulan用に訓練用のデータを作成する．
    PARAM dir_path:訓練データがあるディレクトリパス args:argumentparserの引数
    RETURN 二次元配列  list [tuple (list [unicode ラベル列], list [unicode token])] 
    """
    for label in thompson_training_map:
        for one_instance in thompson_training_map[label]:
            #tokens_in_label=[t for doc in thompson_training_map[label] for t in doc];
            training_data_list.append( (label, one_instance) );
    return training_data_list;

def create_multilabel_datastructure(dir_path, args):
    """
    mulan用に訓練用のデータを作成する．
    PARAM dir_path:訓練データがあるディレクトリパス args:argumentparserの引数
    RETURN 二次元配列  list [tuple (list [unicode ラベル列], list [unicode token])] 
    """
    training_data_list=[];
    level=args.level;
    for fileindex, filepath in enumerate(make_filelist(dir_path)):
        if level==1:
            alphabet_label_list=(os.path.basename(filepath)).split('_')[:-1];
        elif level==2:
            alphabet_label=(os.path.basename(filepath))[0];
        file_obj=codecs.open(filepath, 'r', 'utf-8');
        tokens_in_label=tokenize.wordpunct_tokenize(file_obj.read());
        file_obj.close();
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

def main(level, mode, all_thompson_tree, stop, dutch, thompson, tfidf, exno, args):
    #result_stack=return_range.find_sub_tree(input_motif_no, all_thompson_tree) 
    #print 'The non-terminal nodes to reach {} is {}'.format(input_motif_no, result_stack);
    if mode=='big':
        bigdoc_module.big_doc_main(all_thompson_tree, args);   
    elif mode=='class':
        if level==1:
            construct_classifier_for_1st_layer(all_thompson_tree, stop, dutch, thompson, tfidf, args)


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
    parser.add_argument('-training', help='which training tool? liblinear or mulan?');
    parser.add_argument('-mulan_model', help='which model in mulan library.\
                        RAkEL, RAkELd, MLCSSP, HOMER, HMC, ClusteringBased, Ensemble etc.',
                        default=u'');
    parser.add_argument('-reduce_method', help='which method use to reduce feature dimention?\
                        labelpower, copy, binary',
                        default='binary');
    parser.add_argument('-save_exno', help='not in use', default=u'');
    args=parser.parse_args();
    dir_path='../parsed_json/'
    #------------------------------------------------------------    
    if float(args.training_amount)>=1.0:
        sys.exit('[Warning] -training_amount must be between 0-1(Not including 1)');
    #------------------------------------------------------------    
    if args.easy_domain==True:
        if not (args.dutch==True and args.thompson==True):
            sys.exit('[Warning] You specified easy_domain mode. But there is only one domain');
    #------------------------------------------------------------    
    if args.training=='mulan' and args.mulan_model==u'':
        sys.exit('[Warning] mulan model is not choosen');
    #------------------------------------------------------------    
    if args.mode=='class':
        if args.training==u'mulan': 
            pass;
        elif args.training==u'liblinear':
            pass;
        else:
            sys.exit('[Warning] training tool is not choosen(mulan or liblinear)')        
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
