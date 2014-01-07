# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:33:15 2013

@author: kensuke-mi
"""
import sys, codecs, re;

def convert_to_feature_space(training_map,feature_map_character,feature_map_numeric,tfidf_score_map,tfidf,tfidf_idea,args):
    if tfidf_idea==1:
        easy_domain_flag=False;
    elif tfidf_idea==2:
        easy_domain_flag=False;
    elif tfidf_idea==3:
        easy_domain_flag=False;
    elif tfidf_idea==4:
        easy_domain_flag=True;
    elif tfidf_idea==5:
        easy_domain_flag=False;                                
    
    if args.training=='mulan':                     
        training_data_feature_space=convert_to_feature_space_mulan(training_map,feature_map_character,feature_map_numeric,tfidf_score_map,tfidf, easy_domain_flag, args);
    elif args.training=='liblinear':
        training_data_feature_space=convert_to_feature_space_liblinear(training_map,feature_map_character,feature_map_numeric,tfidf_score_map,tfidf, tfidf_idea, easy_domain_flag, args);

    return training_data_feature_space;

def convert_to_feature_space_arow(training_subdata, feature_map_character,feature_map_numeric,tfidf_score_map,tfidf, easy_domain_flag, args):
    """
    ARGS training_subdata dictionary {unicode keyname:list [unicode token] }
    RETURN training_subdata_featurespace dictionary {unicode keyname:list [tuple (int feature_number, float feature_value)]}
    """
    training_subdata_featurespace={};
    for motiflabel in training_subdata:
        instances_in_label_featurespace=[];        
        #--------------------------------------------------------------- 
        #ラベル内訓練事例を素性空間に変換する           
        for one_instance in training_subdata[motiflabel]:
            one_instance_feature_space=[];
            #---------------------------------------------------------------
            #１訓練事例内のtokenを素性空間に変換する
            #つまりは素性辞書を引いているだけ
            for t in one_instance:
                if t in feature_map_character:
                    for feature_candidate in feature_map_character[t]:
                        feature_number=feature_map_numeric[feature_candidate];
                        if args.tfidf==True:
                            t_weight=tfidf_score_map[t];                            
                            one_instance_feature_space.append( (feature_number, t_weight) );
                        elif args.tfidf==False:
                            one_instance_feature_space.append( (feature_number, 1) );
            #---------------------------------------------------------------
            if not one_instance_feature_space==[]:
                instances_in_label_featurespace.append(one_instance_feature_space);
        #---------------------------------------------------------------            
        training_subdata_featurespace[motiflabel]=instances_in_label_featurespace;    
    return training_subdata_featurespace;
                       
def convert_to_feature_space_liblinear(training_map,feature_map_character,feature_map_numeric,tfidf_score_map,tfidf, tfidf_idea, easy_domain_flag, args):
    exno=args.experiment_no;
    easy_domain2=easy_domain_flag;
    #文字表現の素性を保存する
    character_feature_outfile=codecs.open('../classifier/character_expression/character_expression.'+str(exno), 'w', 'utf-8');
    character_feature_outfile.write('Gold label\tcharacter feature\n')
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
                            #文字表現の素性フォーマットを作成する
                            character_expression_format=u'{}\t{}\n'.format(label, candidate);
                            
                            feature_pattern=re.compile(u'^{}'.format(prefix));
                            if re.search(feature_pattern, candidate):
                                domain_feature=candidate;
                                domain_feature_numeric=feature_map_numeric[domain_feature];
                                if tfidf==False:
                                    feature_space_doc.append((domain_feature_numeric,
                                                              1)); 
                                    character_feature_outfile.write(character_expression_format);
                                #ここがtfidfが真の場合は，素性値をタプルにして追加すればよい
                                elif tfidf==True:
                                    feature_space_doc.append((domain_feature_numeric,
                                                              tfidf_score_map[token]));
                                    character_feature_outfile.write(character_expression_format);
                            
                            #easy domainの一般素性
                            if re.search(ur'^g', candidate):
                                general_feature=candidate;
                                general_feature_numeric=feature_map_numeric[general_feature];
                                if tfidf==False:
                                    feature_space_doc.append((general_feature_numeric,
                                                              1));
                                    character_feature_outfile.write(character_expression_format);
                                #ここがtfidfが真の場合は，素性値をタプルにして追加すればよい
                                elif tfidf==True:
                                    feature_space_doc.append((general_feature_numeric,
                                                              tfidf_score_map[token]));
                                    character_feature_outfile.write(character_expression_format);

                            if easy_domain2==True:
                                label_domain_name=u'{}_{}'.format(label, token);
                                if label_domain_name in candidate:
                                #if re.search(ur'^[A-Z]_', candidate):
                                    domain_feature=candidate;
                                    domain_feature_numeric=feature_map_numeric[domain_feature];
                                    if tfidf==False:
                                        feature_space_doc.append((domain_feature_numeric,
                                                                  1)); 
                                        character_feature_outfile.write(character_expression_format);
                                    #ここがtfidfが真の場合は，素性値をタプルにして追加すればよい
                                    elif tfidf==True:
                                        if token in tfidf_score_map:
                                            feature_space_doc.append((domain_feature_numeric,
                                                                      tfidf_score_map[token]));
                                            character_feature_outfile.write(character_expression_format);
                            
                            if tfidf_idea==3:
                                #idea3は素性ベクトルを定数倍させる
                                if tfidf==True:
                                    if token in tfidf_score_map:
                                        feature_space_doc.append((candidate,
                                                                  tfidf_score_map[token]));
                                        character_feature_outfile.write(character_expression_format);
                                else:
                                    feature_space_doc.append((candidate,
                                                              1));
                                    character_feature_outfile.write(character_expression_format);
                #------------------------------------------------------------     
                #素性空間に射影した文書をlabelごとに管理するmapの下に入れる
                if not feature_space_doc==[]:
                    if label not in feature_space_label:
                        feature_space_label[label]=[feature_space_doc];
                    else:
                        feature_space_label[label].append(feature_space_doc);
            #------------------------------------------------------------
        if not feature_space_label=={}:
            training_map_feature_space[subdata]=feature_space_label;
        #------------------------------------------------------------ 
    character_feature_outfile.close();
    
    return training_map_feature_space;

def convert_to_feature_space_mulan(training_map,feature_map_character,feature_map_numeric,tfidf_score_map, tfidf, easy_domain_flag,args):
    """
    RETURN list [tuple (list [unicode モチーフラベル], list [unicode token])] 
    """
    exno=args.experiment_no;
    #文字表現の素性を保存する
    character_feature_outfile=codecs.open('../classifier/character_expression/character_expression.'+str(exno), 'w', 'utf-8');
    character_feature_outfile.write('Gold label\tcharacter feature\n')
    
    training_data_list_feature_space=[];
    training_data_list=training_map;
    for one_instance in training_data_list:
        one_instance_stack=[];
        motiflabel_stack=one_instance[0];
        #============================================================
        for motiflabel in motiflabel_stack:
            for token in one_instance[1]:
                if args.tfidf==False:
                    if token in feature_map_character:
                        for feature_candidate in feature_map_character[token]:
                            character_format=u'{}\t{}\n'.format(motiflabel, feature_candidate);
                            character_feature_outfile.write(character_format);
                            
                            feature_number=feature_map_numeric[feature_candidate];                                
                            one_instance_stack.append((feature_number, 1));
                elif args.tfidf==True:
                    #ドメイン一致用のキー
                    tokenname_domain=u'{}_{}'.format(motiflabel, token);
                    tokenname_general=u'g_{}'.format(token);
                    if token in tfidf_score_map and token in feature_map_character:
                        for feature_candidate in feature_map_character[token]:
                            #ラベルのeasy domain adoptation専用
                            #ラベルと一致する素性だけを発火させる
                            if tokenname_domain in feature_candidate or tokenname_general in feature_candidate:                                    
                                character_format=u'{}\t{}\n'.format(motiflabel, feature_candidate);
                                character_feature_outfile.write(character_format);
                                
                                feature_number=feature_map_numeric[feature_candidate];
                                tfidf_weight=tfidf_score_map[token];
                                one_instance_stack.append((feature_number, tfidf_weight));
        """
        if easy_domain_flag==True:
            for motiflabel in motiflabel_stack:
                for token in one_instance[1]:
                    if args.tfidf==False:
                        if token in feature_map_character:
                            for feature_candidate in feature_map_character[token]:
                                character_format=u'{}\t{}\n'.format(motiflabel, feature_candidate);
                                character_feature_outfile.write(character_format);
                                
                                feature_number=feature_map_numeric[feature_candidate];                                
                                one_instance_stack.append((feature_number, 1));
                    elif args.tfidf==True:
                        #ドメイン一致用のキー
                        tokenname_domain=u'{}_{}'.format(motiflabel, token);
                        tokenname_general=u'g_{}'.format(token);
                        if token in tfidf_score_map and token in feature_map_character:
                            for feature_candidate in feature_map_character[token]:
                                #ラベルのeasy domain adoptation専用
                                #ラベルと一致する素性だけを発火させる
                                if tokenname_domain in feature_candidate or tokenname_general in feature_candidate:                                    
                                    character_format=u'{}\t{}\n'.format(motiflabel, feature_candidate);
                                    character_feature_outfile.write(character_format);
                                    
                                    feature_number=feature_map_numeric[feature_candidate];
                                    tfidf_weight=tfidf_score_map[token];
                                    one_instance_stack.append((feature_number, tfidf_weight));
        #============================================================
        else:
            for token in one_instance[1]:
                if args.tfidf==False:
                    if token in feature_map_character:
                        for feature_candidate in feature_map_character[token]:
                            character_format=u'{}\t{}\n'.format(motiflabel, feature_candidate);
                            character_feature_outfile.write(character_format);                            
                            
                            feature_number=feature_map_numeric[feature_candidate];
                            one_instance_stack.append((feature_number, 1));
                elif args.tfidf==True:
                    if token in tfidf_score_map and token in feature_map_character:
                        for feature_candidate in feature_map_character[token]:
                            character_format=u'{}\t{}\n'.format(motiflabel, feature_candidate);
                            character_feature_outfile.write(character_format);                            
                            
                            feature_number=feature_map_numeric[feature_candidate];
                            tfidf_weight=tfidf_score_map[token];
                            one_instance_stack.append((feature_number, tfidf_weight));
                            """
        #============================================================
        training_data_list_feature_space.append((one_instance[0], one_instance_stack));
    character_feature_outfile.close();
    return training_data_list_feature_space;
