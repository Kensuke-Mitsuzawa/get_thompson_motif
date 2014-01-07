#! /usr/bin/python
# -*- coding:utf-8 -*-
import subprocess, codecs, sys, argparse, shutil;
import feature_function; 
__date__='2014/01/06';
memory_option=u'-Xmx30g'

def out_to_mulan_format(training_data_list, feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        feature_space, motif_vector, tfidf_idea, args):
    """
    mulan用にデータフォーマットを作成する．
    RETURN void
    """
    exno=args.experiment_no;
    training_data_list_feature_space=feature_function.convert_to_feature_space(training_data_list,
                                                                             feature_map_character,
                                                                             feature_map_numeric,
                                                                             tfidf_score_map, tfidf,
                                                                             tfidf_idea, args);
    #------------------------------------------------------------
    #arffファイルのheader部分を作成
    #xmlファイルも同時に作成
    file_contents_stack=[];
    xml_contents_stack=[];
    file_contents_stack.append(u'@relation hoge\n\n');
    xml_contents_stack.append(u'<?xml version="1.0" encoding="utf-8"?>\n<labels xmlns="http://mulan.sourceforge.net/labels">\n')
    for feature_tuple in sorted(feature_map_numeric.items(), key=lambda x:x[1]):
        file_contents_stack.append(u'@attribute {} numeric\n'.format(feature_tuple[1]));
    for motif_name in motif_vector:
        file_contents_stack.append(u'@attribute {} {{0,1}}\n'.format(motif_name));
        xml_contents_stack.append(u'<label name="{}"></label>\n'.format(motif_name));
    xml_contents_stack.append(u'</labels>');
    file_contents_stack.append(u'\n\n');
    #------------------------------------------------------------
    #時間がかかるため，argsの引数でトレーニングデータの量を管理．ただし，デフォルトの時は，何もしない
    num_training_instances=len(training_data_list_feature_space);
    print 'All training instances is {}'.format(num_training_instances);
    if args.training_amount=='0.95':
        training_amount_limit=num_training_instances;
    else:
        training_amount_limit=int(num_training_instances*args.training_amount);
    #------------------------------------------------------------
    #arffファイルのデータ部分を作成
    debug_l=[];
    file_contents_stack.append(u'@data\n');
    for instance_index, one_instance in enumerate(training_data_list_feature_space):
        feature_space_for_one_instance=[0]*feature_space;
        motif_vector_numeric=[0]*len(motif_vector);       
        for motif in one_instance[0]:
            motif_vector_numeric[motif_vector.index(motif)]=1;
            debug_l.append(motif_vector_numeric);
        for feature_number_tuple in one_instance[1]:
            #素性番号は１始まりに設定しているので，インデックス調整のために-1する必要がある
            feature_space_for_one_instance[feature_number_tuple[0]-1]=feature_number_tuple[1];
        feature_space_for_one_instance=[str(item) for item in feature_space_for_one_instance];
        motif_vector_str=[str(item) for item in motif_vector_numeric];
        file_contents_stack.append(u','.join(feature_space_for_one_instance)\
                                   +u','+u','.join(motif_vector_str)\
                                   +u'\n');
        #limitの上限に達したら打ち切り
        if instance_index==training_amount_limit:
            break;
    file_contents_stack.append(u'\n');
    #------------------------------------------------------------
    output_filepath=u'../classifier/mulan/';
    output_filestem=u'exno{}.arff'.format(exno);
    with codecs.open(output_filepath+output_filestem, 'w', 'utf-8') as f:
        f.writelines(file_contents_stack);
    #------------------------------------------------------------
    output_filestem=u'exno{}.xml'.format(exno);
    with codecs.open(output_filepath+output_filestem, 'w', 'utf-8') as f:
        f.writelines(xml_contents_stack);
    #============================================================ 
    call_mulan(args);

def call_mulan(args):
    model_type=args.mulan_model;
    reduce_method=args.reduce_method;
    xml_train='../classifier/mulan/exno{}.xml'.format(args.experiment_no);
    arff_train='../classifier/mulan/exno{}.arff'.format(args.experiment_no);
    if not args.save_exno==u'':
        model_savepath='../classifier/mulan/exno{}.model'.format(args.save_exno);
        shutil.copy(arff_train, '../classifier/mulan/exno{}.arff'.format(args.save_exno));    
        shutil.copy(xml_train, '../classifier/mulan/exno{}.xml'.format(args.save_exno));    
        shutil.copy('../classifier/feature_map_character_1st.json.{}'.format(args.experiment_no),
                    '../classifier/feature_map_character_1st.json.{}'.format(args.save_exno));
        shutil.copy('../classifier/feature_map_numeric_1st.json.{}'.format(args.experiment_no),
                    '../classifier/feature_map_numeric_1st.json.{}'.format(args.save_exno));
        shutil.copy('../classifier/tfidf_word_weight.json.{}'.format(args.experiment_no),
                    '../classifier/tfidf_word_weight.json.{}'.format(args.save_exno));
    else:
        model_savepath='../classifier/mulan/exno{}.model'.format(args.experiment_no);
    dimention_reduce_method=args.reduce_method; 
    args=('java {} -jar ./mulan_interface/train_classifier_method.jar -arff {} -xml {} -reduce True -reduce_method {} -model_savepath {} -model_type {}'.format(memory_option,arff_train,xml_train,reduce_method,model_savepath,model_type)).split();
    
    print 'Input command is following:{}'.format(u' '.join(args));                                  
    subproc_args = {'stdin': subprocess.PIPE,
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.STDOUT,
                    'close_fds' : True,}
    try:
        p = subprocess.Popen(args, **subproc_args)  # 辞書から引数指定
    except OSError:
        print "Failed to execute command: %s" % args[0];
        sys.exit(1);
    
    output=p.stdout;
    print u'-'*30;
    for line in output:
        print line;

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='');
    #以下のオプションは，「使う素性は同じなんだけど，モデルだけを変えたい，という時に使う」
    parser.add_argument('-experiment_no', help='If you have training file already, specify its exno',
                        required=True); 
    parser.add_argument('-save_exno', help='experiment number to save trained model',
                        required=True);
    parser.add_argument('-mulan_model', help='which model in mulan library.\
                        RAkEL, RAkELd, MLCSSP, HOMER, HMC, ClusteringBased, Ensemble etc.',
                        default=u'');
    parser.add_argument('-reduce_method', help='which method use to reduce feature dimention?\
                        labelpower, copy, binary',
                        default='binary');
    args=parser.parse_args();
    call_mulan(args);
