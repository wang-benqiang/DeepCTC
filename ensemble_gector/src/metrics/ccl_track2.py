# coding:utf-8
# 将预测的目标句子与源句子比较，得到edits
# python pair2edits_char.py $SRC_PATH $HYP_PATH > $OUTPUT_PATH
import os

import Levenshtein


def gen_track2_outfile(src_path, tgt_path, out_file):
    
    out_file = open(out_file, 'w', encoding='utf8')
    with open(src_path) as f_src, open(tgt_path) as f_tgt:
        lines_src = f_src.readlines()
        lines_tgt = f_tgt.readlines()
        # lines_sid = f_sid.readlines()
        assert len(lines_src) == len(lines_tgt)
        for i in range(len(lines_src)):
            id, src_line = lines_src[i].strip().replace(',', '，').split('\t')
            id2, tgt_line = lines_tgt[i].strip().replace(',', '，').split('\t')
            # sid = lines_sid[i].strip().split('\t')[0]

            # edits = Levenshtein.opcodes(src_line, tgt_line)
            # reverse
            _edits = Levenshtein.opcodes(src_line[::-1], tgt_line[::-1])[::-1]
            edits = []
            src_len = len(src_line)
            tgt_len = len(tgt_line)
            for edit in _edits:
                edits.append((edit[0], src_len - edit[2], src_len - edit[1], tgt_len - edit[4], tgt_len - edit[3]))

            # merge coterminous Levenshtein edited spans
            merged_edits = []
            for edit in edits:
                if edit[0] == 'equal':
                    continue
                if len(merged_edits) > 0:
                    last_edit = merged_edits[-1]
                    if last_edit[0] == 'insert' and edit[0] == 'insert' and last_edit[2] == edit[1]:
                        new_edit = ('insert', last_edit[1], edit[2], last_edit[3], edit[4])
                        merged_edits[-1] = new_edit
                    elif last_edit[2] == edit[1]:
                        assert last_edit[4] == edit[3]
                        new_edit = ('hybrid', last_edit[1], edit[2], last_edit[3], edit[4])
                        merged_edits[-1] = new_edit
                    elif last_edit[0] == 'insert' and edit[0] == 'delete' \
                        and tgt_line[last_edit[3]:last_edit[4]] == src_line[edit[1]:edit[2]]:
                        new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                        merged_edits[-1] = new_edit
                    elif last_edit[0] == 'delete' and edit[0] == 'insert':
                        if src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]]:
                        # print(src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]])
                            new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                            merged_edits[-1] = new_edit
                        elif edit[4] < len(tgt_line) and tgt_line[edit[3]] == tgt_line[edit[4]] and src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]+1:edit[4]+1]:
                            new_edit = ('luanxu', last_edit[1], edit[2]+1, last_edit[3], edit[4])
                            merged_edits[-1] = new_edit
                        else:
                            merged_edits.append(edit)
                    else:
                        merged_edits.append(edit)
                else:
                    merged_edits.append(edit)
            merged_edits2 = []
            for edit in merged_edits:
                if edit[0] == 'equal':
                    continue
                if len(merged_edits2) > 0:
                    last_edit = merged_edits2[-1]
                    if last_edit[0] == 'insert' and edit[0] == 'insert' and last_edit[2] == edit[1]:
                        new_edit = ('insert', last_edit[1], edit[2], last_edit[3], edit[4])
                        merged_edits2[-1] = new_edit
                    elif last_edit[2] == edit[1]:
                        assert last_edit[4] == edit[3]
                        new_edit = ('hybrid', last_edit[1], edit[2], last_edit[3], edit[4])
                        merged_edits2[-1] = new_edit
                    elif last_edit[0] == 'insert' and edit[0] == 'delete' \
                        and tgt_line[last_edit[3]:last_edit[4]] == src_line[edit[1]:edit[2]]:
                        new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                        merged_edits2[-1] = new_edit
                    elif last_edit[0] == 'delete' and edit[0] == 'insert':
                        if src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]]:
                        # print(src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]:edit[4]])
                            new_edit = ('luanxu', last_edit[1], edit[2], last_edit[3], edit[4])
                            merged_edits2[-1] = new_edit
                        elif edit[4] < len(tgt_line) and tgt_line[edit[3]] == tgt_line[edit[4]] and src_line[last_edit[1]:last_edit[2]] == tgt_line[edit[3]+1:edit[4]+1]:
                            new_edit = ('luanxu', last_edit[1], edit[2]+1, last_edit[3], edit[4])
                            merged_edits2[-1] = new_edit
                        else:
                            merged_edits2.append(edit)
                    else:
                        merged_edits2.append(edit)
                else:
                    merged_edits2.append(edit)
            # generate edit sequence
            result = []
            for edit in merged_edits2:
                if tgt_line[edit[3]:edit[4]] == '[UNK]':
                    continue
                if edit[0] == "insert":
                    result.append((str(edit[1]+1), str(edit[1]+1), "M", tgt_line[edit[3]:edit[4]]))
                elif edit[0] == "replace":
                    # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
                    result.append((str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
                elif edit[0] == "delete":
                    result.append((str(edit[1]+1), str(edit[2]), "R"))
                elif edit[0] == "hybrid":
                    # new_op = post_process_S(src_line, (str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
                    result.append((str(edit[1]+1), str(edit[2]), "S", tgt_line[edit[3]:edit[4]]))
                elif edit[0] == "luanxu":
                    result.append((str(edit[1]+1), str(edit[2]) , "W"))

            # print
            # out_line = id +',\t'
            if result:
                for res in result:
                    out_file.write(id + ',\t'+',\t'.join(res) + '\n')
                    # print(id + ',\t'+',\t'.join(res))
            else:
                out_file.write(id+',\tcorrect'+'\n')
                # print(id+',\tcorrect')
        out_file.close()
        


def evaluate_track2_test(src_path, 
                         tgt_path, 
                         out_file, 
                         ref_file, 
                         report_fp):
    
    gen_track2_outfile(src_path, tgt_path, out_file)
    cmd_line = 'perl src/metrics/evaluation.pl {} {} {}'.format(out_file, report_fp, ref_file)
    rtn = os.system(cmd_line)
    print(rtn)



if __name__ == '__main__':
    src_fp = 'data/ccl_2022/samples/track2/src.txt'
    trg_fp = 'data/ccl_2022/samples/track2/hyp.txt'
    out_fp = 'logs/output.txt'
    ref_fp = 'data/ccl_2022/samples/track2/ref.txt'
    report_fp = 'logs/ref.txt'
    evaluate_track2_test(src_fp, trg_fp, out_fp, ref_fp, report_fp)
    