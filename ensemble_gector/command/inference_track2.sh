cd ..
python -m src.corrector.corrector_ctc \
--gector_dir "model/cltc2/gector" \
--test_fp "tests/data/ccl2022_cltc/track2/cged_test.txt" \
--out_fp "logs/gector/track2_cged_pred.txt" \
--out_res_fp "logs/ensemble/cged_pred1.txt"
