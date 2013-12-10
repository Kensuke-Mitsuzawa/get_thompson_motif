#! /usr/bin/python
# -*- coding:utf-8 -*-

import os, glob, sys;
import random, shutil;

def make_filelist(dir_path):
    file_list=[];
    for root, dirs, files in os.walk(dir_path):
        for f in glob.glob(os.path.join(root, '*')):
            file_list.append(f);
    return file_list;

def copy_files(target_dir, filelist):
    for file_i in filelist:
        copy_to_path=target_dir+os.path.basename(file_i);
        shutil.copyfile(file_i, copy_to_path);

def split_train_test():
    training_ratio=0.99;
    test_ratio=1-training_ratio;
    #------------------------------------------------------------
    #dir_path='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated/';
    dir_path='../dutch_folktale_corpus/dutch_folktale_database_query_translated_google/';
    #save_dir_train='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated_train/';
    save_dir_train='../dutch_folktale_corpus/dutch_folktale_database_query_translated_google_translated_train/';
    #save_dir_test='../dutch_folktale_corpus/dutch_folktale_database_google_translated/translated_test/';
    save_dir_test='../dutch_folktale_corpus/dutch_folktale_database_query_translated_google_translated_test/';
    try:
        shutil.rmtree(save_dir_train);
        shutil.rmtree(save_dir_test);
    except OSError:
        pass;
    try:
        os.makedirs(save_dir_train);
    except:
        pass;
    try:
        os.makedirs(save_dir_test);
    except:
        pass;
    #------------------------------------------------------------
    filelist=make_filelist(dir_path);
    num_of_files=len(filelist);
    num_for_train=int(num_of_files*training_ratio);
    num_for_test=num_of_files-num_for_train;

    random.shuffle(filelist);
    filelist_for_train=filelist[:num_for_train];
    filelist_for_test=filelist[num_for_train:];
    copy_files(save_dir_train, filelist_for_train);
    copy_files(save_dir_test, filelist_for_test);

if __name__=='__main__':
    split_train_test();
