# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:14:12 2013

@author: kensuke-mi
素性作成方法をアイディアごとに，関数へ切り分けしている
"""
#__date__="2013/12/26"
import codecs
import construct_bigdoc_or_classifier, tf_idf;
from nltk import tokenize;
from nltk import stem;
from nltk.corpus import stopwords;
lemmatizer = stem.WordNetLemmatizer();


lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];

def create_tfidf_feat_idea1(training_map, feature_map_character, args):
    import math;
    stop=args.stop;
    L2_flag=True;
    persian_flag=False;
    #------------------------------------------------------------
    #TFIDFスコアはthompson resourceとdutch_folktale_corpusとtest documentのすべてを合わせた空間で求める
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
    if persian_flag==True:
        #ペルシア語口承文芸コーパスからファイルを読み込む
        test_corpus_instances=[];
        persian_folktale_documet_path='../../corpus_dir/translated_google/'
        for doc_filepath in construct_bigdoc_or_classifier.make_filelist(persian_folktale_documet_path):
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
        all_training_instances=all_training_instances+test_corpus_instances;
    #------------------------------------------------------------
    print 'TFIDF(Idea-1) score calculating'
    print 'L2 flag:{} Persian flag:{}'.format(L2_flag, persian_flag);
    tfidf_score_map=tf_idf.tf_idf_test(all_training_instances);
    #------------------------------------------------------------
    if L2_flag==True:
        #L2正則化をかける        
        weight_sum=0;    
        for key in tfidf_score_map:
            weight=tfidf_score_map[key];
            weight_sum+=(weight)**2;
        L2_norm=math.sqrt(weight_sum);
        L2_normalized_map={};    
        L2_weightsum=0;
        for key in tfidf_score_map:
            normalized_score=tfidf_score_map[key]/L2_norm;
            L2_normalized_map[key]=normalized_score;        
            #L2で正規化した重みの和を求める
            L2_weightsum+=normalized_score;
        #足切りスコアの算出
        L2_average=L2_weightsum/len(L2_normalized_map);
        for doc in all_training_instances:
            for t in doc:
                if t in L2_normalized_map:
                    #足切り
                    if L2_normalized_map[t] > L2_average:
                        pass;
                    else:
                        weight_format=u'{}_{}_{}'.format('normal', t, L2_normalized_map[t]);
                        if t not in feature_map_character:
                            feature_map_character[t]=[weight_format];
                        elif weight_format not in feature_map_character[t]:
                            feature_map_character[t].append(weight_format);
    """ 
    feature_map_character=make_tfidf_feature_from_score(tfidf_score_map,
                                                        wordset_map,
                                                        feature_map_character, args);
    """
    return feature_map_character, tfidf_score_map;
    
def create_tfidf_feat_idea2_4(training_map, feature_map_character, tfidf_idea, args):
    easy_domain_flag=args.easy_domain2;
    if tfidf_idea==2:
        #素性ベクトルを定数倍しない
        constant=True;
        #g_ ではじまる素性は作らない
        easy_domain_flag=False;
    elif tfidf_idea==3:
        #素性ベクトルを定数倍する
        constant=True;
        #g_ で始まる素性は作らない
        easy_domain_flag=True;
    elif tfidf_idea==4:
        #素性ベクトルを定数倍する
        constant=True;
        #g_ で始まる素性は作る
        easy_domain_flag=True;
    elif tfidf_idea==5:
        #素性ベクトルを定数倍する
        constant=True;
        #g_ で始まる素性は作る
        easy_domain_flag=False;        
    
    feature_map_character, L2_normalized_map=create_tfidf_feat_idea_general(training_map, feature_map_character, constant, easy_domain_flag, args);
            
    return feature_map_character, L2_normalized_map;

def create_tfidf_feat_idea_general(training_map, feature_map_character, constant, easy_domain_flag, args):
    #ラベルごとの重要語を取り出す
    #TFIDFスコアを文書集合から算出した後，ラベル文書ごとに閾値（足切り値）を求め，閾値以下の語は素性を作らない
    #これで，「あるラベルに特徴的な語」を示す素性が作れた．と思う
    import math;
    bigdocs_stack=[];
    bigdocs_stack_nolabel=[];
    print 'Add general feature for label domain adaptation?:{}'.format(easy_domain_flag);
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
    print 'TFIDF(Idea-2,3,4) score calculating'
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
        denominator=0;
        for t in tokens:
            if t in L2_normalized_map:
                word_weight=L2_normalized_map[t];
                weightsum_of_label+=word_weight;
                denominator+=1;
        #閾値を，重み平均に設定する(中央値でも良い気がする)
        #12/23 この行は間違いだと思う．だって，ラベルあたりの重み合計を，L2_normalized_mapで割ったらダメじゃん
        #いま知りたいのは，ラベルごとの重み平均．なので，正しい分母は「ラベル内に存在している異なり単語数」
        #threshold_point=weightsum_of_label/len(L2_normalized_map); 
        threshold_point=weightsum_of_label/len(L2_normalized_map); 
        for t in tokens:            
            if t in L2_normalized_map:
                #対数化しているので，符号が逆になる
                if L2_normalized_map[t] > threshold_point:  
                    pass;
                else:
                    #定数倍した時と，そうでない時に差があるのか？を検証するため
                    if constant==True:
                        weight_format=u'{}_{}_{}'.format(label, t, L2_normalized_map[t]);
                    else:
                        weight_format=u'normal_{}_{}'.format(t, L2_normalized_map[t]);
                    if t not in feature_map_character:
                        feature_map_character[t]=[weight_format];
                    elif weight_format not in feature_map_character[t]:
                        feature_map_character[t].append(weight_format);
                    #------------------------------------------------------------ 
                    #ラベル素性の他に，general素性を入れてみる 
                    if easy_domain_flag==True:
                        weight_format=u'{}_{}_{}'.format('g', t, L2_normalized_map[t]);
                        if t not in feature_map_character:
                            feature_map_character[t]=[weight_format];
                        elif weight_format not in feature_map_character[t]:
                            feature_map_character[t].append(weight_format);

    return feature_map_character, L2_normalized_map;

