#! /usr/bin/python
# -*- coding:utf-8 -*-
import subprocess, codecs, sys;
from construct_bigdoc_or_classifier import convert_to_feature_space;

def out_to_mulan_format(training_data_list, feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        feature_space, motif_vector, args):
    """
    mulan用にデータフォーマットを作成する．
    RETURN void
    """
    exno=args.experiment_no;
    training_data_list_feature_space=convert_to_feature_space(training_data_list,
                                                            feature_map_character,
                                                            feature_map_numeric,
                                                            tfidf_score_map, tfidf, args);
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
    file_contents_stack.append(u'@data\n');
    for instance_index, one_instance in enumerate(training_data_list_feature_space):
        feature_space_for_one_instance=[0]*feature_space;
        motif_vector_numeric=[0]*len(motif_vector);
        for motif in one_instance[0]:
            motif_vector_numeric[motif_vector.index(motif)-1]=1;
        for feature_number_tuple in one_instance[1]:
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
    xml_train='../classifier/mulan/exno{}.xml'.format(args.experiment_no);
    arff_train='../classifier/mulan/exno{}.arff'.format(args.experiment_no);
    model_savepath='../classifier/mulan/exno{}.model'.format(args.experiment_no);
    args=('java -jar ./mulan_interface/train_meta_classifier_method.jar -arff {} -xml {} -reduce True -model_savepath {} -model_type {}'.format(arff_train,xml_train,model_savepath,model_type)).split();
    
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
    args="";
    call_mulan(args);
