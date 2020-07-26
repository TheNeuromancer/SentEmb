for model in 'binwang/bert-base-uncased' 'binwang/roberta-base' 'binwang/xlnet-base-cased' 'binwang/bert-base-nli' 'binwang/bert-base-nli-stsb' 'binwang/bert-large-nli' 'binwang/bert-large-nli-stsb' 'USE'

do
    for method in dissecting ave_last_hidden CLS ave_one_layer
    do
#        echo "$model\_$method.txt"
#        echo "python 4_sent_comp.py --sent-file active_sentences.txt --model_type $model --embed_method $method | tee results_active/$model\_$method.txt"
         python 4_sent_comp.py --sent-file active_sentences.txt --model_type $model --embed_method $method | tee results_active/$model\_$method.txt
    done
done