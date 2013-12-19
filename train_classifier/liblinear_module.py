#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Thu Dec 12 12:16:52 2013

@author: kensuke-mi
"""
import sys, codecs;
#change below by an environment
libsvm_wrapper_path='/home/kensuke-mi/opt/libsvm-3.17/python/';
#libsvm_wrapper_path='/Users/kensuke-mi/opt/libsvm-3.17/python/';
#liblinear_wrapper_path='/Users/kensuke-mi/opt/liblinear-1.94/python/';
#sys.path.append(liblinear_wrapper_path);
sys.path.append(libsvm_wrapper_path);
from liblinearutil import *;
from svmutil import *;
import scale_grid;

def close_test(classifier_path, test_path):
    print 'close test result for {} with {}'.format(classifier_path, test_path);
    y, x = svm_read_problem(test_path);
    m = load_model(classifier_path);
    p_label, p_acc, p_val = predict(y, x, m);
    ACC, MSE, SCC = evaluations(y, p_label); 
    print ACC, MSE, SCC;

def unify_tarining_feature_space(training_map_feature_space):
    unified_map={};
    for subdata_key in training_map_feature_space:
        for label in training_map_feature_space[subdata_key]:
            if label not in unified_map:
                unified_map[label]=training_map_feature_space[subdata_key][label];
            else:
                unified_map[label]+=training_map_feature_space[subdata_key][label];
    return unified_map;

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


