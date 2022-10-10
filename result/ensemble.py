

def ensembel_results(res_fp_list, out_fp):
    crt_ct = 3
    edit_ct = 3
    if isinstance(res_fp_list, str):
        res_fp_list = list(res_fp_list)

    detect_edits, correction_edits = {}, {}

    model_num = len(res_fp_list)
    for res_fp in res_fp_list:
        for line in open(res_fp, 'r', encoding='utf-8'):
            line_res = line.strip().replace(',', '').split()
            if len(line_res) == 5:
                # S, M
                sent_id, start_idx, end_idx, error_type, words = line_res
                key = (start_idx, end_idx, error_type)
                if sent_id not in detect_edits:
                    detect_edits[sent_id] = {}
                if sent_id not in correction_edits:
                    correction_edits[sent_id] = {}

                if key in detect_edits[sent_id]:

                    detect_edits[sent_id][key] += 1
                else:
                    detect_edits[sent_id][key] = 1

                crt_key = (start_idx, end_idx, error_type, words)

                if crt_key in correction_edits[sent_id]:
                    correction_edits[sent_id][crt_key] += 1
                else:
                    correction_edits[sent_id][crt_key] = 1

            elif len(line_res) == 4:
                sent_id, start_idx, end_idx, error_type = line_res
                key = (start_idx, end_idx, error_type)
                if sent_id not in detect_edits:
                    detect_edits[sent_id] = {}
                if key in detect_edits[sent_id]:
                    detect_edits[sent_id][key] += 1
                else:
                    detect_edits[sent_id][key] = 1
            elif len(line_res) == 2:
                sent_id, right_str = line_res
                key = (sent_id, right_str)
                if sent_id not in detect_edits:
                    detect_edits[sent_id] = {}
                if key in detect_edits[sent_id]:
                    detect_edits[sent_id][key] += 1
                else:
                    detect_edits[sent_id][key] = 1

    final_edits = {}
    for sent_id, edits in detect_edits.items():
        if (sent_id, 'correct') in edits and edits[(sent_id, 'correct')] >= crt_ct:
            final_edits[sent_id] = ['correct']
        else:
            final_edits[sent_id] = []
            for detect_key, value in edits.items():
                if len(detect_key) >= edit_ct and value >= edit_ct:
                    (start_idx, end_idx, error_type) = detect_key
                    if error_type.upper() in ('R', 'W'):
                        final_edits[sent_id].append(detect_key)
                    else:
                        for crt_key, crt_value in sorted(correction_edits[sent_id].items(), key=lambda x: x[1], reverse=True):
                            if (crt_key[0], crt_key[1], crt_key[2]) == (start_idx, end_idx, error_type) and crt_value >= crt_ct:
                                final_edits[sent_id].append(crt_key)
                                break

    out_fp = open(out_fp, 'w', encoding='utf8')

    for sent_id, edit_list in final_edits.items():

        if len(edit_list) == 0 or (len(edit_list) == 1 and edit_list[0] == 'correct'):
            line = '{},\t{}\n'.format(sent_id, 'correct')
        else:
            line = ''
            for edit in edit_list:
                if len(edit) >= 3:
                    _line = '{},\t'.format(sent_id)
                    _line += ',\t'.join(list(edit))+'\n'
                    line += _line
        out_fp.write(line)

    out_fp.close()
    return final_edits


def ensemble2(base_fp, other_fp, out_fp):

    correction_edits = {}
    for line in open(base_fp, 'r', encoding='utf8'):
        line_res = line.strip().replace(',', '').split()

        if len(line_res) == 5:
            # S, M
            sent_id, start_idx, end_idx, error_type, words = line_res
            key = (start_idx, end_idx, error_type, words)
            if sent_id not in correction_edits:
                correction_edits[sent_id] = [key]
            else:
                correction_edits[sent_id].append(
                    key)

        elif len(line_res) == 4:
            sent_id, start_idx, end_idx, error_type = line_res
            key = (start_idx, end_idx, error_type)
            if sent_id not in correction_edits:
                correction_edits[sent_id] = [key]
            else:
                correction_edits[sent_id].append(
                    key)

        elif len(line_res) == 2:
            sent_id, right_str = line_res
            key = right_str
            if sent_id not in correction_edits:
                correction_edits[sent_id] = [key]
            else:
                correction_edits[sent_id].append(
                    key)

    append_correction_edits = {}
    for line in open(other_fp, 'r', encoding='utf8'):
        line_res = line.strip().replace(',', '').split()

        if len(line_res) == 5:
            # S, M
            sent_id, start_idx, end_idx, error_type, words = line_res
            key = (start_idx, end_idx, error_type, words)
            if sent_id not in append_correction_edits:
                append_correction_edits[sent_id] = [key]
            else:
                append_correction_edits[sent_id].append(
                    key)

        elif len(line_res) == 4:
            sent_id, start_idx, end_idx, error_type = line_res
            key = (start_idx, end_idx, error_type)
            if sent_id not in append_correction_edits:
                append_correction_edits[sent_id] = [key]
            else:
                append_correction_edits[sent_id].append(
                    key)

        elif len(line_res) == 2:
            sent_id, right_str = line_res
            key = right_str
            if sent_id not in append_correction_edits:
                append_correction_edits[sent_id] = [key]
            else:
                append_correction_edits[sent_id].append(
                    key)

    
    ct = 0
    ct1 = 0
    filter_num = 0
    for sent_id, edits in correction_edits.items():
        if edits[0] == 'correct':
            if append_correction_edits[sent_id][0] != 'correct':
                if len(append_correction_edits[sent_id]) > 1:
                    # 
                    correction_edits[sent_id] = append_correction_edits[sent_id]
                # print(sent_id, correction_edits[sent_id])
                    ct+=1
                    
                elif len(append_correction_edits[sent_id]) == 1:
                    if append_correction_edits[sent_id][0][-2] in ('S'):
                        if append_correction_edits[sent_id][0][-2] == 'S' and append_correction_edits[sent_id][0][-1] in ('他', '她','它'):
                            print('filter',sent_id,append_correction_edits[sent_id])
                            filter_num+=1
                            
                        else:
                            correction_edits[sent_id] = append_correction_edits[sent_id]
                # print(sent_id, correction_edits[sent_id])
                        ct+=1
                else:
                    print('1 edit',sent_id,append_correction_edits[sent_id])
                    ct1+=1
    
    # 移除他她它
    ta_num = 0
    import copy
    for sent_id, edit_list in correction_edits.items():
        
        bak = copy.deepcopy(edit_list)
        
        correction_edits[sent_id] = [edit for edit in edit_list if not (len(edit)==4 and edit[-1] in ('他', '她','它') and edit[-2]=='S')]
        if len(correction_edits[sent_id]) == 0:
            correction_edits[sent_id] = ['correct']
        if correction_edits[sent_id] != bak:
            print('{}, origin:{}, new:{}'.format(sent_id, bak, correction_edits[sent_id]))
            ta_num +=1
    
    out_fp = open(out_fp, 'w', encoding='utf8')

    for sent_id, edit_list in correction_edits.items():

        if len(edit_list) == 0 or (len(edit_list) == 1 and edit_list[0] == 'correct'):
            line = '{},\t{}\n'.format(sent_id, 'correct')
        else:
            line = ''
            for edit in edit_list:
                if len(edit) >= 3:
                    _line = '{},\t'.format(sent_id)
                    _line += ',\t'.join(list(edit))+'\n'
                    line += _line
        out_fp.write(line)

    out_fp.close()
    print('final ct:{}, ct1:{}, filter_num:{}, ta_num{}'.format(ct, ct1, filter_num, ta_num))
    print('end')
if __name__ == '__main__':
    # fp_list = [
    #     # 'logs/cged.pred.txt',
    #     # 'logs/t5/cged.pred.txt',
    #     # 'logs/ensemble/4583.txt',
    #     'logs/gector_copy/cged.pred.txt',
    #     'logs/gector/cged.pred.txt',
    #     'logs/gector/cged.pred1.txt',
    #     'logs/cged.pred.txt',
    #     'logs/ensemble/cged.pred_ddd.txt',
    #     'logs/ensemble/4583.txt',
    #     'logs/ensemble/cged.pred.txt'

    # ]
    # # out_list = []
    # r = ensembel_results(fp_list, out_fp='logs/ensemble/cged.pred333.txt')
    # print(r)

    ensemble2(
        base_fp='cged.pred_1.txt',
        other_fp='cged.pred_1model_3.txt',
        out_fp='cged.pred.txt'
    )
