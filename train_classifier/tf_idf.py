#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__='Kensuke Mitsuzawa';
__data__='2013/12/01';
import nltk, json, codecs, sys, re, os, glob, math;
from collections import Counter;

def IDF(term_i, total_document, file_num_having_query):
    numerator=total_document;
    denominator=file_num_having_query[term_i];
    IDF_score=math.log(numerator/denominator);
    return IDF_score;

def TF(term_i, document_j, token_frequency_dict):
    denominator=len(document_j);
    numerator=token_frequency_dict[term_i];
    return float(numerator)/denominator;

def tf_idf_test(docs):  
    #docs=[[token.encode('ascii') for token in doc] for doc in docs];
    documents=[];
    for one_doc in docs:
        doc=[]; 
        for t in one_doc:
            try:
                doc.append(t.encode('ascii'));
            except UnicodeEncodeError:
                pass;
        documents.append(doc);
    tf_idf_score={};
    tokens = [];  
    for doc in documents:  
        tokens += doc  
    A = nltk.TextCollection(documents)  
    token_types = set(tokens)  
    for token_type in token_types:
        if not A.tf_idf(token_type, tokens)==0: 
            tf_idf_score[token_type]=math.log(A.tf_idf(token_type, tokens));
        else:
            tf_idf_score[token_type]=0;
    return tf_idf_score;

def main(document_set):
    tf_idf_score={};
    doc_num_having_query={};
    all_token=[];
    [all_token.append(t) for document in document_set for t in document];
    all_token=list(set(all_token));
    for token in all_token:
        for document in document_set:
            if token in document:
                if token in doc_num_having_query:
                    doc_num_having_query[token]+=1;
                else:
                    doc_num_having_query[token]=1;
    total_document=len(document_set);
    for document_index, document_j in enumerate(document_set):
        token_frequency_dict=Counter(document_j);
        for term_index, term_i in enumerate(document):
            w_ij=IDF(term_i, total_document, doc_num_having_query)*TF(term_i, document_j, token_frequency_dict);
            tf_idf_score[term_i]=w_ij;
    return tf_idf_score;
if __name__=='__main__':
    document_set=["The demands from Beijing have resulted in tensions with Japan and the United States.",
                  "On Saturday, United, American and Delta airlines told CNN that its pilots were following Washington's advice and complying with Beijing's 'air defense identification zone."];
    document_set=[sentence.split() for sentence in document_set];
    #tf_idf_score=main(document_set);
    tf_idf_score=tf_idf_test(document_set);
