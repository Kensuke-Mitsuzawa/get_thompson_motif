#! /usr/bin/pytho
# -*- coding:utf-8 -*-

import sys, codecs, re, lxml.html, os, json;


def parse(html_file_path):
    middle_node_attribute={'align': 'center', 'style': 'text-align:center;'};
    html=open(html_file_path, 'rb').read();
    root=lxml.html.fromstring(html);

    motif_map_stack=[];
    first_layer_map={};
    second_layer_map={};
    third_layer_map={};
    fourth_layer_map={};
    second_layer_name=u'';
    third_layer_name=u'';
    p_node_list=root.findall('p') 
    for p_node in p_node_list:
        if p_node.attrib==middle_node_attribute:
            child_node_list=list(p_node);
            if child_node_list==[]:
                pass;
            #======================================== 
            #新しい２層目のノードの獲得，新たに３層目ノードの獲得開始
            elif child_node_list[0].tag=='span':
                #古い３層目のマップを２層目のマップに保存
                second_layer_map[second_layer_name]=third_layer_map;
                #新しい２層目の名前を獲得 
                second_layer_name=child_node_list[0].text;
                #３層目のマップを初期化
                third_layer_map={};
            #======================================== 
            #非常にイレギュラーであるが，３層目がこうやって表現されることがある
            #<b><span> </b>
            elif child_node_list[0].tag=='b':
                if list(child_node_list[0])==[]:
                    pass;
                elif list(child_node_list[0])[0].tag=='span':
                    #古い四層目を３層目に保存
                    third_layer_map[((child_node_list[0])[0].text)]=fourth_layer_map;
                    #新しい４層目キー名を取得
                    third_layer_name=((child_node_list[0])[0].text);
                    #TODO ファイル構造的にかなりまずいキー名なのでなんとかすること
                    #４層目のマップを初期化
                    fourth_layer_map={};
            #======================================== 
            #TODO イレギュラーでない４層目の獲得方法
        #======================================== 
        #葉にあたる部分の要素獲得
        if not p_node.text==None:
            fourth_layer_label_name=p_node.text;
            if re.search(ur'[A-Z].+', fourth_layer_label_name):
                label_ID=(fourth_layer_label_name.split(u'\r\n')[0]);
                leaf_layer_map={}; 
                #iノード（たぶん概要）の獲得
                if not p_node.find('i')==None:
                    fourth_layer_outline=p_node.find('i').text;
                    #iノードの後ろ（資料情報）の獲得
                    if not (p_node.find('i').tail)==None:
                        example_material=p_node.find('i').tail
                    else:
                        example_material=None; 
                else:
                    fourth_layer_outline=None;
                    example_material=None;

            #４層目マップへの要素の追加
            leaf_layer_map[label_ID]={'label_name':fourth_layer_label_name,\
                                      'label_outline':fourth_layer_outline,\
                                      'example_material':example_material}; 
            fourth_layer_map[label_ID]=leaf_layer_map;
        #TODO A0のようなタイプのノードへの対応
        #葉の要素獲得ここまで
        #======================================== 
    print second_layer_map

if __name__=='__main__':
    path='./a.htm'
    parse(path);
