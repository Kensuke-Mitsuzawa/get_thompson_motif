#! /usr/bin/pytho
# -*- coding:utf-8 -*-

import sys, codecs, re, lxml.html, os, json;

def numeric_process(middle_label):
    if re.search(ur'.+\d+--.+\d+\. .+\r\n.+$', middle_label):
        numeric, char=re.split(ur'\. ', middle_label);
        numeric=numeric.replace(u'\u2020', u'');
        char=char.replace(u'\r\n', u' ');
        range_start, range_end=numeric.split(u'--');
        prefix=range_start[0];
        range_start=re.sub(ur'^[A-Z]', u'',range_start);
        range_end=re.sub(ur'^[A-Z]', u'',range_end);
        #(頭文字, 開始番号，終了番号)
        range_tuple=(prefix, range_start, range_end, char)
        
    elif re.search(ur'.+\d+--.+\d+\.\r\n.+$', middle_label):
        numeric, char=middle_label.split(u'\r\n');
        numeric=numeric.replace(u'\u2020', u'');
        char=char.replace(u'\r\n', u' ');
        numeric=numeric.strip(u'.')
        range_start, range_end=numeric.split(u'--');
        prefix=range_start[0];
        range_start=re.sub(ur'^[A-Z]', u'',range_start);
        range_end=re.sub(ur'^[A-Z]', u'',range_end);
        #(頭文字, 開始番号，終了番号)
        range_tuple=(prefix, range_start, range_end, char);

    #U.htmlの場合に変なフォーマットがあったので，それに対処するため
    elif re.search(ur'.+\d+-.+\d+\.\s.+\r\n.+--.+', middle_label):
        numeric, char=middle_label.rstrip(u'.').split(u'.');
        numeric=numeric.replace(u'\u2020', u'');
        char=(char.replace(u'\r\n', u' ')).strip();
        numeric=numeric.strip(u'.')
        range_start, range_end=numeric.split(u'-');
        prefix=range_start[0];
        range_start=re.sub(ur'^[A-Z]', u'',range_start);
        range_end=re.sub(ur'^[A-Z]', u'',range_end);
        #(頭文字, 開始番号，終了番号)
        range_tuple=(prefix, range_start, range_end, char);

    elif re.search(ur'.+\d+--.+\d+\..*', middle_label):
        numeric=middle_label.replace(u'\u2020', u'');
        range_start, range_end=numeric.split(u'--');
        prefix=range_start[0];
        range_start=re.sub(ur'^[A-Z]', u'',range_start);
        range_end=re.sub(ur'^[A-Z]', u'',range_end);
        range_start=range_start.strip();
        #問題がおきたからテスト的に書き換え
        #range_end=range_end.strip().strip(u'.');
        range_end=range_end.strip().split(u'.')[0];
        range_tuple=(prefix, range_start, range_end, None);       
    
    return range_tuple;

def construct_classifier(html_file_path):
    #htmlページは構成がめちゃくちゃであり，そのままでは分類の木が正しく作れない．そこで，まずあらかじめ数字の範囲だけを取得してしまう．
    middle_node_attribute={'align': 'center', 'style': 'text-align:center;'};
    html=open(html_file_path, 'rb').read();
    root=lxml.html.fromstring(html);

    p_node_list=root.findall('p');

    range_list=[];
    for p_node in p_node_list:
        range_tuple=None;
        if p_node.attrib==middle_node_attribute:
            #U.htmlに対しては，まったく獲得できていないので，ここ以下の問題かと思う
            if not p_node.text==None:
                middle_label=p_node.text;
                if re.search(ur'.+\d+--.+\d+\. .+\r\n.+$', middle_label):
                    range_tuple=numeric_process(middle_label)
                elif re.search(ur'.+\d+--.+\d+\.\r\n.+$', middle_label):
                    range_tuple=numeric_process(middle_label)
                elif re.search(ur'.+\d+--.+\d+\..*', middle_label):
                    range_tuple=numeric_process(middle_label)
                    
            else: 
                child_node_list=list(p_node);
                for child_node in child_node_list:
                    #たぶん，これで小分類が獲得できているはず
                    if not child_node.text==None:
                        middle_label=child_node.text;
                        if re.search(ur'.+\d+--.+\d+\. .+\r\n.+$', middle_label):
                            range_tuple=numeric_process(middle_label)
                        elif re.search(ur'.+\d+--.+\d+\.\r\n.+$', middle_label):
                            range_tuple=numeric_process(middle_label)
                        elif re.search(ur'.+\d+-+.+\d+\..+', middle_label):
                            range_tuple=numeric_process(middle_label)
                    #<b><span> <b>のケースに対処するため
                    grandchild_node_list=list(child_node);
                    for grandchild_node in grandchild_node_list:
                        middle_label=grandchild_node.text;
                        if re.search(ur'.+\d+--.+\d+\. .+\r\n.+$', middle_label):
                            range_tuple=numeric_process(middle_label)
                        elif re.search(ur'.+\d+--.+\d+\.\r\n.+$', middle_label):
                            range_tuple=numeric_process(middle_label)
        if not range_tuple==None:
            range_list.append(range_tuple);
    return range_list;

def sort_classifier(range_list):
    #まずはすべてのキーに対して，マップを用意
    range_map_in_map={};
    for item in range_list:
        range_map_in_map.setdefault(item, {});

    deletion_key_stack=[];
    for compare_item_1 in range_map_in_map:
        for compare_item_2 in range_map_in_map:
            if not compare_item_1==compare_item_2:
                #２つの辞書を比較してTrue(compare_item_2がitem1の範囲内だったら，item2をitem1の下に入れる)
                if compare_range(compare_item_1, compare_item_2)==True:
                    range_map_in_map[compare_item_1].setdefault(compare_item_2, {});
                    deletion_key_stack.append(compare_item_2);
                    #new_range_map_in_map.setdefault()
    try:
        for item in deletion_key_stack:
            del range_map_in_map[item];
    except:
        pass;

    return range_map_in_map;

def compare_range(compare_item_1, compare_item_2):
    item1_start_range=int(compare_item_1[1]);
    item1_end_range=int(compare_item_1[2]);

    item2_start_range=int(compare_item_2[1]);
    item2_end_range=int(compare_item_2[2]);

    if item2_start_range >= item1_start_range and item2_end_range <= item1_end_range:
        #print item1_start_range, item1_end_range
        #print item2_start_range, item2_end_range
        #print 'item2 is in the range of item_1';
        #print '===================='
        return True;
    else:
        return False;

def parse(html_file_path):
    middle_node_attribute={'align': 'center', 'style': 'text-align:center;'};
    html=open(html_file_path, 'rb').read();
    root=lxml.html.fromstring(html);

    """
    motif_map_stack=[];
    first_layer_map={};
    second_layer_map={};
    third_layer_map={};
    fourth_layer_map={};
    second_layer_name=u'';
    third_layer_name=u'';
    """
    leaf_tuple_stack=[];
    p_node_list=root.findall('p') 
    for p_node in p_node_list:
        """
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
                    #コメントアウトの部分，間違って２層目を取得してる．．
                    '''
                    #古い四層目を３層目に保存
                    third_layer_map[((child_node_list[0])[0].text)]=third_layer_map;
                    #新しい４層目キー名を取得
                    third_layer_name=((child_node_list[0])[0].text);
                    #TODO ファイル構造的にかなりまずいキー名なのでなんとかすること
                    #４層目のマップを初期化
                    third_layer_map={};
                    '''
                elif list(child_node_list[0])[0].tag=='span':
                    #古い四層目を３層目に保存
                    third_layer_map[((child_node_list[0])[0].text)]=fourth_layer_map;
                    #新しい４層目キー名を取得
                    third_layer_name=((child_node_list[0])[0].text);
                    #TODO ファイル構造的にかなりまずいキー名なのでなんとかすること
                    #４層目のマップを初期化
                    third_layer_map={};
                    """
        #======================================== 
        #葉にあたる部分の要素獲得
        if not p_node.text==None:
            fourth_layer_label_name=p_node.text;
            if re.search(ur'[A-Z].+', fourth_layer_label_name):
                label_ID=(fourth_layer_label_name.split(u'\r\n')[0]);
                leaf_layer_map={}; 
                #iノード（たぶん概要）の獲得
                if not p_node.find('i')==None:
                    #この時点でu.htmlにはiノードがないケースがたくさん
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
            leaf_tuple_stack.append( (fourth_layer_label_name,\

                                    fourth_layer_outline,\
                                    example_material) ); 
        #TODO A0のようなタイプのノードへの対応
        #葉の要素獲得ここまで
        #======================================== 
    return leaf_tuple_stack;

def insertion_leaf_2_tree(range_map_in_map, leaf_tuple_stack):
    leaf_parent_map={};
    #葉の階層の情報から辞書を作成する
    for leaf_tuple in leaf_tuple_stack:
        numeric_expression=leaf_tuple[0];
        if re.search(ur'^[A-Z][0-9]+', leaf_tuple[0].strip()) and re.search(ur'-+', leaf_tuple[0].strip())==None:
            class_number=leaf_tuple[0].strip().split()[0];
            class_number=re.sub(ur'^[A-Z]', u'', class_number);
            #==============================
            #class_numberがleaf階層で親（まだ下に子どもを持っていれば）だったら辞書に登録
            if len(class_number.split(u'.'))==2:
                leaf_parent_map[class_number.split(u'.')[0]]={'content':leaf_tuple,\
                                                              'child':[]};
            #==============================
            #class_numberがすでに辞書に登録ずみの子どもだったら，childのリストにタプルを追加
            else:
                parent_number=class_number.split(u'.')[0]
                if not parent_number in leaf_parent_map:
                    class_number=parent_number.rstrip(u'.');
                    leaf_parent_map[class_number]={'content':leaf_tuple,\
                                                    'child':[]};
                else:
                    (leaf_parent_map[parent_number]['child']).append(leaf_tuple);

    #葉の階層情報を分類して登録する
    copied_range_map_in_map=range_map_in_map.copy();
    for item in leaf_parent_map:
        leaf_parent_number=int(item);
        #tree_node_numberはタプルで構成される cf.(u'B', u'200', u'299', None)
        for tree_node_number in range_map_in_map:
            range_start=int(tree_node_number[1]);
            range_end=int(tree_node_number[2]);
            #モチーフ番号が当てはまる階層を探して，当てはまる範囲に登録する
            if leaf_parent_number >= range_start and leaf_parent_number <=range_end:
                if copied_range_map_in_map[tree_node_number]=={}:
                    copied_range_map_in_map[tree_node_number]={item:leaf_parent_map[item]};
                else:
                    tuple_in_flag=False;
                    for already_element in copied_range_map_in_map[tree_node_number]:
                        #この階層がキーである可能性があるので
                        #現状ではこれより下の階層にキーがないと仮定する（そう思いたい）
                        if isinstance(already_element, tuple):
                            range_child_start=int(already_element[1]);
                            range_child_end=int(already_element[2]);
                            if leaf_parent_number >= range_child_start and leaf_parent_number <= range_end:
                                tuple_in_flag=True;
                                if copied_range_map_in_map[tree_node_number][already_element]=={}:
                                    copied_range_map_in_map[tree_node_number][already_element]={item:leaf_parent_map[item]}
                                else:
                                    copied_range_map_in_map[tree_node_number][already_element].setdefault(item, leaf_parent_map[item]);
                    #この階層にタプルがなくて，かつタプルがあったとしても範囲でなかった場合 
                    if tuple_in_flag==False:
                        (copied_range_map_in_map[tree_node_number]).setdefault(item, leaf_parent_map[item]);
                         
    return copied_range_map_in_map;

def draw_tree(range_map_in_map):
    first_layer_stack=[];
    print u'=============================='
    print u'ROOT';
    for top_item in range_map_in_map:
        #first_layer_stack.append(u'{}_{}_{}'.format(top_item[1]. top_item[2], top_item[3]));
        print u'|-- {}_{}_{}'.format(top_item[1], top_item[2], top_item[3])

def re_construct_map(original_map):
    """
    現状の辞書（キーがタプル）はjsonに出力できないので，アンダースコア接続にキーを書き換える
    """
    re_constructed={};
    for keyname in original_map:
        new_keyname=u'{0}_{1}_{2}_{3}'.format(keyname[0],
                                              keyname[1],
                                              keyname[2],
                                              keyname[3]);
        re_constructed[new_keyname]={};
        for second_keyname in original_map[keyname]:
            if isinstance(second_keyname, tuple):
                new_second_keyname=u'{0}_{1}_{2}_{3}'.format(second_keyname[0],
                                                             second_keyname[1],
                                                             second_keyname[2],
                                                             second_keyname[3]);
                re_constructed[new_keyname][new_second_keyname]={};
                for third_keyname in original_map[keyname][second_keyname]:
                    new_third_keyname=third_keyname
                    if isinstance(third_keyname, tuple):
                        print 'Still another level exists:{}'.format(third_keyname);
                    else:
                        re_constructed[new_keyname][new_second_keyname].setdefault(new_third_keyname,\
                                                                                   original_map\
                                                                                   [keyname]\
                                                                                   [second_keyname]\
                                                                                   [third_keyname]);
            else:
                new_second_keyname=second_keyname;
                re_constructed[new_keyname].setdefault(new_second_keyname, original_map[keyname][second_keyname])
            
    #デバッグ用に残しておく
            """
    for level1 in re_constructed:
        print 'level 1 keyname:{}'.format(level1)
        for level2 in re_constructed[level1]:
            print 'level 2 keyname:{}'.format(level2);
    """
        return re_constructed;

if __name__=='__main__':
    path=sys.argv[1];
    range_list=construct_classifier(path);
    range_map_in_map=sort_classifier(range_list); 
    leaf_tuple_stack=parse(path);
    range_map_in_map=insertion_leaf_2_tree(range_map_in_map, leaf_tuple_stack);
    draw_tree(range_map_in_map);
    
    re_constructed=re_construct_map(range_map_in_map);
    filename=os.path.basename(path);
    with codecs.open('./parsed_json/'+filename+'.json', 'w', 'utf-8') as f:
        json.dump(re_constructed, f, indent=4, ensure_ascii=False);
