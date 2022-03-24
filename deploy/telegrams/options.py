from os import path

import fairseq

import logzero
from logzero import logger


def create_args():
    parser = fairseq.options.get_interactive_generation_parser()
    parser = parser.add_argument_group("Dialogues")
    parser.add_argument('--api_token', type=str, metavar="ID", required=True, help="API token")
    parser.add_argument('--json_args', type=path.abspath, metavar="FP", help="Path to args file (json format)")
    parser.add_argument('--yaml_args', type=path.abspath, metavar="FP", help="Path to args file (yaml format)")
    parser.add_argument('--spm', type=path.abspath, metavar="FP", help="Path to sentencepiece model")
    parser.add_argument('--max_len_contexts', type=int, metavar="N", default=10, help="The maximum length of contexts")
    parser.add_argument('--start_message', type=str, metavar="TEXT", default="こんにちは", help="The first chat message")
    parser.add_argument('--device_fairseq', type=int, metavar="N", help="Cuda device of fairseq model")
    parser.add_argument('--log_prefix', type=path.abspath, metavar="FP", default="log.txt", help="Path to log output")
    parser.add_argument('--max_turn', type=int, metavar="N", default=30, help="The maximum dialogue turn")

    parser = parser.add_argument_group("Socket")
    parser.add_argument('--host_address', type=str, default='127.0.0.1', help='host_address')

    parser = parser.add_argument_group("Topic Extractor")
    parser.add_argument('--vector_file', type=path.abspath, metavar="FP", help="Path to fasttext vector file")
    parser.add_argument('--category_file', type=path.abspath, metavar="FP", help="Path to wikipedia category file")
    parser.add_argument('--word_freq_file', type=path.abspath, metavar="FP", help="Path to word frequency file")
    parser.add_argument('--word_freq_min_threshold', default=1000, type=int, metavar="N", help="Lowest word frequency")
    parser.add_argument('--word_freq_max_threshold', default=10000, type=int, metavar="N", help="Highest word frequency")
    parser.add_argument('--top_n_topic', default=1, type=int, metavar="N", help="Number of top candidates to be sampled")
    parser.add_argument('--cot_prob', default=1, type=float, metavar="N", help="Probability to change topic (COT)")
    parser.add_argument('--min_cot_contexts', default=10, type=int, metavar="N", help="Minimum context length for COT")

    parser = parser.add_argument_group("Next Utterance Predictor")
    parser.add_argument('--mlm_model', type=path.abspath, metavar="FP", help="Path to (masked) language model params.")
    parser.add_argument('--device_mlm', type=int, metavar="N", help="Cuda device of mlm")

    parser = parser.add_argument_group("Postprocess Scorer")
    parser.add_argument('--sif_param', type=float, default=1e-3, metavar="N", help='SIF weighting param')
    parser.add_argument('--init_wordfreq_fname', type=path.abspath, metavar='FP',
                       help='Path to word frequency file to compute inf score')
    parser.add_argument('--inf_thr', type=float, default=3., metavar="N",
                       help='Cands with lower inf_score than this thr are eliminated')
    parser.add_argument('--jac_thr', type=float, default=0.3, metavar="N",
                       help='Cands with higher jac_score than this thr are eliminated')
    parser.add_argument('--user_jac_thr', type=float, default=1., metavar="N",
                       help='Set this thr if you have to filter out cands considering jac_score with user utterances')
    parser.add_argument('--mix_jac_thr', type=float, default=0.2, metavar="N",
                       help='Set this thr if you have to filter out cands considering jac_score with all contexts')
    parser.add_argument('--duplicate_score_thr', type=float, default=0.55, metavar="N",
                       help='Threshold. The more duplicate words the sentence has, the smaller this score.')
    parser.add_argument('--next_uttr_thr', type=float, metavar="N", default=0.99,
                       help="Threshold of next sentence prediction score")
    parser.add_argument('--select_by_information', action='store_true',
                       help="If true, the response was chosen to have the highest 'inf' score from accepted responses.")
    parser.add_argument('--incremental_thr_value', type=float, metavar="N", default=0.05,
                       help="The value for gradually increasing the threshold")
    parser.add_argument('--jsnli_model_fname', type=path.abspath, metavar='FP',
                       help='Path to jsnli model file to compute jsnli score')
    parser.add_argument('--filter_by_jsnli', action='store_true',
                       help="If true, Cands was filtered out by jsnli score from accepted responses.")
    parser.add_argument('--filter_by_prev_jsnli', action='store_true',
                       help="If true, Cands was filtered out by prev jsnli score from accepted responses.")


    parser = parser.add_argument_group("Wikipedia Rule")
    parser.add_argument('--wiki_knowledge', type=path.abspath, metavar="FP",
                       help='File on wikipedia knowledges for knowledge dialogue')
    parser.add_argument('--wiki_response_template', type=path.abspath, metavar="FP",
                       help='File on response templates for knowledge dialogue')
    parser.add_argument('--wiki_template', type=path.abspath, metavar="FP",
                       help='File on templates for knowledge dialogue')

    parser = parser.add_argument_group('NTTCS_model')
    parser.add_argument('--nttcs_port', type=int, default=40000)

    # parser = parser.add_argument_group('Other Fairseq model')
    # parser.add_argument('--other_json_args', type=path.abspath, default="/work02/SLUD2021/models/ilys_aoba_bot/args_bot_telegram_run.json")
    # parser.add_argument('--other_data', type=path.abspath, default="/work02/SLUD2021/models/ilys_aoba_bot/fairseq_vocab")
    parser = parser.add_argument_group('Our Fairseq model')
    parser.add_argument('--our_fairseq_port', type=int, default=42000)

    parser = parser.add_argument_group('DialoGPT')
    parser.add_argument('--dgpt_model', type=path.abspath, default=f'/work02/SLUD2021/models/dialogpt/GP2-pretrain-step-100000.pkl')
    parser.add_argument('--dgpt_config', type=path.abspath, default=f'/work02/SLUD2021/models/dialogpt/config.json')
    parser.add_argument('--dgpt_toker', type=path.abspath, default=f'/work02/SLUD2021/models/dialogpt/tokenizer')
    parser.add_argument('--dgpt_max_history', type=int, default=1)
    parser.add_argument('--dgpt_port', type=int, default=45000)
    
    parser = parser.add_argument_group('DPR')
    parser = parser.add_argument('--dpr_config', type=path.abspath, default='/work02/SLUD2021/models/dpr/interact_retriever.yaml')

    parser = parser.add_argument_group('FiD')
    parser = parser.add_argument('--fid_config', type=path.abspath, default='/work02/SLUD2021/models/fid/config.json')
    parser = parser.add_argument('--fid_port', type=int, default=50000)

    parser = parser.add_argument_group('Knowledge Filtering')
    parser = parser.add_argument('--topic_idf_file', type=path.abspath, default='/work02/SLUD2021/datasets/wikipedia/dialogu_idf/dialogue_idf.csv')

    args = fairseq.options.parse_args_and_arch(parser)
    logzero.logfile(args.log_prefix)
    logger.info(args)

    return args
