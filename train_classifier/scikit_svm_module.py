# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:22:33 2013

@author: kensuke-mi
"""
from sklearn import svm;
from sklearn.metrics import classification_report;
from sklearn.cross_validation import train_test_split;
from sklearn.svm import LinearSVC;
from sklearn.metrics import confusion_matrix;
from scipy.sparse import lil_matrix

def training_with_scikit():
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